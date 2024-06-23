from diffusers import StableDiffusionPipeline
import torch
import os
from utils_gpt import chat_with_gpt
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--init_prompt', type=str)
parser.add_argument('--attri_p', type=str)
parser.add_argument('--attri_n', type=str)
parser.add_argument('--n_p', type=int)
parser.add_argument('--get_description', type=int, default=0)

parser.add_argument('--model_path', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--iter', type=int, default=4)
parser.add_argument('--output_path', type=str, default="")
args = parser.parse_args()

SEED = args.seed
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Get target/non-target descriptions by GPT
basename = args.model_path.split("/")[-1]
if args.get_description == 1:
    print("----loading GPT and generate descriptions----")
    init_prompt = args.init_prompt
    attri_p = args.attri_p
    attri_n = args.attri_n
    n_p = args.n_p

    target_description = chat_with_gpt(init_prompt, attri_p, n_p)
    with open(f"./descriptions/{basename}/target_des.txt", 'w') as file:
        for sentence in target_description:
            file.write(sentence + '\n')
            
    nontar_description = chat_with_gpt(init_prompt, attri_n, n_p)  # need to change the cnc into sks
    with open(f"./descriptions/{basename}/nontar_des.txt", 'w') as file:
        for sentence in nontar_description:
            result = re.sub(r'cnc', 'sks', sentence, flags=re.IGNORECASE)
            file.write(result + '\n')
else:
    with open(f"./descriptions/{basename}/target_des.txt", 'r') as file:
        target_description = [line.strip() for line in file]
    with open(f"./descriptions/{basename}/nontar_des.txt", 'r') as file:
        nontar_description = [line.strip() for line in file]

# Generating augmented data
def run(pipe, descriptions, model_id, pseudo):
    for prompt in descriptions:
        for i in range(args.iter):
            image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, height=512, width=512).images[0]
            prompt_out = prompt.replace(" ", "-")
            prompt_out = prompt_out.replace("sks", pseudo)

            if args.output_path == "":
                output_path = "output"
            else:
                output_path = args.output_path
            if not os.path.exists(f"{model_id}/{output_path}/{pseudo}"):
                os.makedirs(f"{model_id}/{output_path}/{pseudo}")
            image.save(f"{model_id}/{output_path}/{pseudo}/{str(i)}_{SEED}_{prompt_out}.png")

torch.manual_seed(SEED)
pipeline = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16, safety_checker=None,
                                                requires_safety_checker=False,).to("cuda:0")

run(pipeline, target_description, args.model_path, "neg")
run(pipeline, nontar_description, args.model_path, "pos")