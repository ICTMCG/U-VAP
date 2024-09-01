import os 
import numpy as np
import argparse
from PIL import Image
import torch
import shutil
import re

import torch
import os
import numpy as np
from torchvision.transforms import ToPILImage
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import open_clip

to_pil = ToPILImage()

def get_best_text(args, image_list):
    '''
    init_prompt: init prompt template
    image_list：original image list
    '''
    # load template
    with open(f"./attributes/{args.attri_n}.txt", 'r') as file:
        neg_templates = [line.strip() for line in file]
    with open(f"./attributes/{args.attri_p}.txt", 'r') as file:
        pos_templates = [line.strip() for line in file]

    init_input_list = []
    init_prompt_list = args.init_prompt.split()
    neg_idx = init_prompt_list.index(args.attri_n) - 1
    pos_idx = init_prompt_list.index(args.attri_p) - 1
    for neg_temp in neg_templates:
        for pos_temp in pos_templates:
            init_input_list.append(args.init_prompt.replace(init_prompt_list[neg_idx], neg_temp).replace(init_prompt_list[pos_idx], pos_temp))

    batch_size = 400
    i = 0
    score_list = []
    while True:
        if i + batch_size < len(init_input_list):
            text = tokenizer(init_input_list[i:i+batch_size]).to(device)
        else:
            text = tokenizer(init_input_list[i:]).to(device)
        gen_batch = [clip_preprocess(i).unsqueeze(0) for i in image_list]
        gen_batch = torch.concatenate(gen_batch).to(device)

        gen_feat = clip_model.encode_image(gen_batch)
        text_feat = clip_model.encode_text(text)
        
        gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
        
        score = gen_feat @ text_feat.t()
        score = torch.mean(score, dim=0)
        score_list = score_list + score.cpu().detach().numpy().tolist()

        del score, text_feat
        
        i += 400
        if i >= len(init_input_list):
            break

    best_init_prompt = init_input_list[np.argmax(score_list)]
    best_init_prompt_list = best_init_prompt.split()
    templates = {args.attri_n: best_init_prompt_list[neg_idx], args.attri_p: best_init_prompt_list[pos_idx]}

    return templates, best_init_prompt

def get_text_img_similarity(init_input, generated_image):
    text = tokenizer([init_input]).to(device)

    gen_batch = [clip_preprocess(i).unsqueeze(0) for i in [generated_image]]
    gen_batch = torch.concatenate(gen_batch).to(device)

    gen_feat = clip_model.encode_image(gen_batch)
    text_feat = clip_model.encode_text(text)
    
    gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)
    text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)

    return (gen_feat @ text_feat.t()).mean().item()

def get_batch_img_similarity(oir_image_list, generated_image):
    gen_batch = [clip_preprocess(i).unsqueeze(0) for i in [generated_image]]
    gen_batch = torch.concatenate(gen_batch).to(device)
    ori_gen_batch = [clip_preprocess(i).unsqueeze(0) for i in oir_image_list]
    ori_gen_batch = torch.concatenate(ori_gen_batch).to(device)

    gen_feat = clip_model.encode_image(gen_batch)
    ori_gen_feat = clip_model.encode_image(ori_gen_batch)
    
    gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)
    ori_gen_feat = ori_gen_feat / ori_gen_feat.norm(dim=1, keepdim=True)

    return (gen_feat @ ori_gen_feat.t()).mean().item()

def get_img_similarity(oir_image, generated_image):
    gen_batch = [clip_preprocess(i).unsqueeze(0) for i in [generated_image]]
    gen_batch = torch.concatenate(gen_batch).to(device)
    ori_gen_batch = [clip_preprocess(i).unsqueeze(0) for i in [oir_image]]
    ori_gen_batch = torch.concatenate(ori_gen_batch).to(device)

    gen_feat = clip_model.encode_image(gen_batch)
    ori_gen_feat = clip_model.encode_image(ori_gen_batch)
    
    gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)
    ori_gen_feat = ori_gen_feat / ori_gen_feat.norm(dim=1, keepdim=True)

    return (gen_feat @ ori_gen_feat.t()).mean().item()

def get_direction_similarity(source_prompt, target_prompt, source_image, target_image, direction_loss):
    with torch.no_grad():
        source_text = tokenizer([source_prompt]).to(device)
        source_text_feat = clip_model.encode_text(source_text)
        target_text = tokenizer([target_prompt]).to(device)
        target_text_feat = clip_model.encode_text(target_text)
        text_direction = (target_text_feat - source_text_feat).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)

        source_batch = [clip_preprocess(i).unsqueeze(0) for i in source_image]
        source_batch = torch.concatenate(source_batch).to(device)
        target_batch = [clip_preprocess(i).unsqueeze(0) for i in [target_image]]
        target_batch = torch.concatenate(target_batch).to(device)
        source_img_feat = clip_model.encode_image(source_batch)
        target_img_feat = clip_model.encode_image(target_batch)
        img_direction = (target_img_feat - source_img_feat)
        
        if img_direction.sum() == 0:
            target_img_feat = clip_model.encode_image(target_batch + 1e-6)
            img_direction = (target_img_feat - source_img_feat)
        img_direction /= (img_direction.clone().norm(dim=-1, keepdim=True))
    
    return 1 - direction_loss(text_direction, img_direction).mean().item()

class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse':    torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae':    torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
        
        return self.loss_func(x, y)

direction_loss = DirectionLoss(loss_type="cosine")

def get_k_img(img_list, score_list, n_k):
    highest_scores = sorted(zip(img_list, score_list), key=lambda x: x[1], reverse=True)[:n_k]
    highest_img, highest_scores = zip(*highest_scores)
    highest_img = list(highest_img)
    highest_scores = list(highest_scores)

    return highest_img, highest_scores


def get_top_k_rank_img(image_paths, img_score_list, text_score_list, n_k):
    rankings_img = sorted(range(len(img_score_list)), key=lambda x: img_score_list[x], reverse=True)
    rankings_text = sorted(range(len(text_score_list)), key=lambda x: text_score_list[x], reverse=True)

    # compute ranking
    rank_sum = [0] * len(image_paths)
    rank_img = [0] * len(image_paths)
    rank_text = [0] * len(image_paths)
    for i, path_index in enumerate(rankings_img):
        rank_sum[path_index] += i
        rank_img[path_index] += i
    for i, path_index in enumerate(rankings_text):
        rank_sum[path_index] += i
        rank_text[path_index] += i

    # data curation
    lowest_indices_raw = sorted(range(len(rank_sum)), key=lambda x: rank_sum[x])  # sorting
    threshold = len(lowest_indices_raw) // 2

    lowest_indices = []
    for item in lowest_indices_raw:
        if rankings_img[item] - rankings_text[item] < threshold:
            lowest_indices.append(item)
        else:
            print("The ranking gap is too large.", image_paths[item], abs(rankings_img[item] - rankings_text[item]))
        if len(lowest_indices) == n_k:
            break

    lowest_paths = [image_paths[i] for i in lowest_indices]
    lowest_img_score_list = [img_score_list[i] for i in lowest_indices]
    lowest_text_score_list = [text_score_list[i] for i in lowest_indices]

    return lowest_paths, lowest_img_score_list, lowest_text_score_list

device = torch.device("cuda")

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--init_prompt', type=str)
parser.add_argument('--ori_path', type=str)
parser.add_argument('--image_path', type=str)
parser.add_argument('--attri_p', type=str)
parser.add_argument('--attri_n', type=str)
parser.add_argument('--output_path', type=str, default="")
parser.add_argument('--n_k', type=int)
args = parser.parse_args()

# load clip
clip_model_name = "ViT-L-14"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained="openai", device=device, jit=True)
tokenizer = open_clip.get_tokenizer(clip_model_name)

# key: The attribute to be replaced; value: Pseudo to be keeped
pseudo_dict = {
    args.attri_n: {"pseudo": "pos", "opposite": args.attri_p}, 
    args.attri_p: {"pseudo": "neg", "opposite": args.attri_n},
    }

# load original image
ori_image_names = []
ori_image_list = []
for i in os.listdir(args.ori_path):
    img = os.path.join(args.ori_path, i)
    ori_image_names.append(img)
    ori_image = Image.open(img).convert('RGB')
    ori_image_list.append(ori_image)

# get best text to describe the ori image
templates, best_init_prompt = get_best_text(args, ori_image_list, pseudo_dict)
print("best word to describe the ori image", templates)


for attri in pseudo_dict:
    pseudo = pseudo_dict[attri]["pseudo"]
    opposite = pseudo_dict[attri]["opposite"]
    path = f"{args.image_path}/{pseudo}"
    new_path = f"{args.image_path}/{pseudo}_filtered"
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
    os.makedirs(new_path)

    # load image
    ori_image_names = []
    ori_image_list = []
    for i in os.listdir(args.ori_path):
        img = os.path.join(args.ori_path, i)
        ori_image_names.append(img)
        ori_image = Image.open(img).convert('RGB')
        ori_image_list.append(ori_image)

    # filtering by image simi
    img_score_list = []
    img_list = []
    for i in os.listdir(path):
        img_path = os.path.join(path, i)
        img = Image.open(img_path).convert('RGB')
        name = i.split('.')[0]
        score = 0
        score += get_batch_img_similarity(ori_image_list, img)  # img simi
        
        # print(img_path, "img score", score)
        img_list.append(img_path)
        img_score_list.append(score)

    # filtering by text simi
    text_score_list = []
    init_prompt_list = args.init_prompt.split()
    neg_idx = init_prompt_list.index(args.attri_n) - 1
    pos_idx = init_prompt_list.index(args.attri_p) - 1
    for img_path in img_list:
        img = Image.open(img_path).convert('RGB')
        name = i.split('.')[0]
        score = 0

        prompt = img_path.split("_")[-1].replace("-", " ").split(".")[0]  # a photo of lamp object in pos color
        prompt = prompt.replace(pseudo, templates[opposite])
        print("prompt", prompt)
        
        score += get_text_img_similarity(prompt, img)
        score -= get_text_img_similarity(best_init_prompt, img)
        
        # print(img_path, "text score", score)
        text_score_list.append(score)

    # sorting
    highest_img, highest_img_scores, highest_text_scores = get_top_k_rank_img(img_list, img_score_list, text_score_list, args.n_k)
    print("highest_img", highest_img)
    print("highest_img_scores", highest_img_scores)
    print("highest_text_scores", highest_text_scores)

    for img_path in highest_img:
        target_img_path = f"{new_path}/{img_path.split('/')[-1]}"
        shutil.copy(img_path, target_img_path)