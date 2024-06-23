from diffusers import StableDiffusionPipeline
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import CLIPTextModel
import torch

import random
import argparse

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPTextModelOutput
from diffusers.loaders import TextualInversionLoaderMixin

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--placeholder_token', type=str)
parser.add_argument('--pos_token', type=str)
parser.add_argument('--neg_token', type=str)
parser.add_argument('--weight', type=float)
parser.add_argument('--num_image', type=int)
parser.add_argument('--seed', type=str)
args = parser.parse_args()

device = torch.device("cuda:0")

def get_pipe_image(pipeline, prompt, placeholder_token, pos_token, neg_token, num_inference_steps=50, guidance_scale=7.5, height=512, width=512):
    # finding placeholder(pseudo)
    if placeholder_token in prompt:
        placeholder_position = prompt.split().index(placeholder_token)
    else:
        placeholder_position = -1

    if isinstance(pipeline, TextualInversionLoaderMixin):
        prompt = pipeline.maybe_convert_prompt(prompt, pipeline.tokenizer)
        pos_token = pipeline.maybe_convert_prompt(pos_token, pipeline.tokenizer)
        neg_token = pipeline.maybe_convert_prompt(neg_token, pipeline.tokenizer)

    text_inputs = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(pipeline.device)
    text_input_ids = text_inputs.input_ids

    if hasattr(pipeline.text_encoder.config, "use_attention_mask") and pipeline.text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    prompt_embeddings = pipeline.text_encoder.text_model.embeddings(input_ids=text_input_ids, position_ids=None)

    # change pseudo embedding
    if placeholder_position != -1:
        special_id = placeholder_position + 1

        b, n, device = *text_input_ids.shape, text_input_ids.device
        pos_text_inputs = pipeline.tokenizer(
            [pos_token],
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(pipeline.device)
        pos_embeddings = pipeline.text_encoder.text_model.embeddings(input_ids=pos_text_inputs.input_ids, position_ids=None)
        neg_text_inputs = pipeline.tokenizer(
            [neg_token],
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(pipeline.device)
        neg_embeddings = pipeline.text_encoder.text_model.embeddings(input_ids=neg_text_inputs.input_ids, position_ids=None)

        new_embeddings = (pos_embeddings[:, 1] + args.weight * (pos_embeddings[:, 1] - neg_embeddings[:, 1])) #/ args.weight
        prompt_embeddings[0, special_id, :] = new_embeddings[0]

    text_embeddings = pipeline.text_encoder(hidden_states=prompt_embeddings, input_ids=text_input_ids, attention_mask=attention_mask)[0]

    image = pipeline(prompt_embeds=text_embeddings, num_inference_steps=50, guidance_scale=7.5, height=512, width=512).images[0]
    
    return image

placeholder_token = args.placeholder_token
pos_token = args.pos_token
neg_token = args.neg_token

prompt = args.prompt
model_id = args.model_path

SEED = args.seed
torch.manual_seed(SEED)
print(model_id)
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16,safety_checker=None,
                                                requires_safety_checker=False,).to(device)

image = get_pipe_image(pipe, prompt, placeholder_token, pos_token, neg_token, num_inference_steps=100, guidance_scale=7.5, height=768, width=768)
# image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, height=512, width=512).images[0]
prompt_out = prompt.replace(" ", "-")

if not os.path.exists(f"{model_id}/output"):
    os.makedirs(f"{model_id}/output")
image.save(f"{model_id}/output/{SEED}_{prompt_out}.png")
