from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from diffusers.loaders import TextualInversionLoaderMixin

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str)
parser.add_argument('--prompt_ori', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--placeholder_token', type=str)
parser.add_argument('--pos_token', type=str)
parser.add_argument('--neg_token', type=str)
parser.add_argument('--weight', type=float)
parser.add_argument('--omega_a', type=float)
parser.add_argument('--omega', type=float)
parser.add_argument('--iter', type=int, required=False, default=None)
parser.add_argument('--seed', type=str)
args = parser.parse_args()

device = torch.device("cuda:0")

def tokenizer(pipeline, prompt):
    text_inputs = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(pipeline.device)
    text_input_ids = text_inputs.input_ids

    return text_input_ids

def encode_prompt(pipeline, text_embeddings, num_images_per_prompt, text_encoder_lora_scale):
    prompt_embeds = pipeline._encode_prompt(
            None,
            device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            prompt_embeds=text_embeddings,
            negative_prompt_embeds=None,
            lora_scale=text_encoder_lora_scale,
        )
    return prompt_embeds

def image_generation_pipeline(pipeline, ori_text_embeddings, text_embeddings, uncond_embeddings, 
                              num_inference_steps=50, guidance_scale_omega=7.5, guidance_scale_omega_a=7.5, num_images_per_prompt=1,
                              height=512, width=512,
                              pos_text_embeddings=None, neg_text_embeddings=None):
    # Define parameters
    height = height or pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    width = width or pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    latents = None
    batch_size = text_embeddings.shape[0]
    device = pipeline._execution_device

    # Encode input prompt
    text_encoder_lora_scale = None
    prompt_embeds = encode_prompt(pipeline, text_embeddings, num_images_per_prompt, text_encoder_lora_scale)
    ori_prompt_embeds = encode_prompt(pipeline, ori_text_embeddings, num_images_per_prompt, text_encoder_lora_scale)
    null_prompt_embeds = encode_prompt(pipeline, uncond_embeddings, num_images_per_prompt, text_encoder_lora_scale)

    if pos_text_embeddings is not None:
        pos_prompt_embeds = encode_prompt(pipeline, pos_text_embeddings, num_images_per_prompt, text_encoder_lora_scale)
    if neg_text_embeddings is not None:
        neg_prompt_embeds = encode_prompt(pipeline, neg_text_embeddings, num_images_per_prompt, text_encoder_lora_scale)

    if pos_text_embeddings is None and neg_text_embeddings is None:
        prompt_embeds = torch.cat([null_prompt_embeds, prompt_embeds])
    else:
        # run adjustment
        prompt_embeds = torch.cat([null_prompt_embeds, ori_prompt_embeds, pos_prompt_embeds, neg_prompt_embeds])

    # Prepare timesteps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    # Prepare latent variables
    generator = None
    eta = 0.0
    num_channels_latents = pipeline.unet.config.in_channels
    latents = pipeline.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)
    
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * prompt_embeds.size(0))
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = pipeline.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            return_dict=False,
        )[0]
        
        # perform guidance
        if prompt_embeds.size(0) == 2:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale_omega_a * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred_uncond, noise_pred_text, pos_pred_text, neg_pred_text = noise_pred.chunk(4)
            
            # latent adjustment and conditional generation
            noise_pred_text_adjust = pos_pred_text + args.weight * (pos_pred_text - neg_pred_text)
            noise_pred = noise_pred_uncond + guidance_scale_omega * (noise_pred_text - noise_pred_uncond) + \
                guidance_scale_omega_a * (noise_pred_text_adjust - noise_pred_text)

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
    output_type = "pil"
    if not output_type == "latent":
        image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    
    do_denormalize = [True] * image.shape[0]
    image = pipeline.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
    
    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None).images[0]
    

def get_pipe_image(pipeline, prompt_ori, prompt, placeholder_token, pos_token, neg_token, num_inference_steps=50, 
                   guidance_scale_omega=7.5, guidance_scale_omega_a=7.5, height=512, width=512):
    if placeholder_token in prompt:
        placeholder_position = prompt.split().index(placeholder_token)
        pos_prompt = prompt.split()
        pos_prompt[placeholder_position] = pos_token
        pos_prompt = ' '.join(pos_prompt)
        neg_prompt = prompt.split()
        neg_prompt[placeholder_position] = neg_token
        neg_prompt = ' '.join(neg_prompt)
    else:
        placeholder_position = -1  # no placeholder_token

    if isinstance(pipeline, TextualInversionLoaderMixin):
        prompt = pipeline.maybe_convert_prompt(prompt, pipeline.tokenizer)
        if placeholder_position != -1:
            pos_prompt = pipeline.maybe_convert_prompt(pos_prompt, pipeline.tokenizer)
            neg_prompt = pipeline.maybe_convert_prompt(neg_prompt, pipeline.tokenizer)

    text_input_ids = tokenizer(pipeline, prompt)
    ori_text_input_ids = tokenizer(pipeline, prompt_ori)

    if placeholder_position != -1:
        pos_text_input_ids = tokenizer(pipeline, pos_prompt)
        neg_text_input_ids = tokenizer(pipeline, neg_prompt)

    text_embeddings = pipeline.text_encoder(text_input_ids)[0]
    ori_text_embeddings = pipeline.text_encoder(ori_text_input_ids)[0]
    if placeholder_position != -1:
        pos_text_embeddings = pipeline.text_encoder(pos_text_input_ids)[0]
        neg_text_embeddings = pipeline.text_encoder(neg_text_input_ids)[0]
    else:
        pos_text_embeddings = None
        neg_text_embeddings = None

    # null-text
    uncond_text_input_ids = tokenizer(pipeline, "")
    uncond_embeddings = pipeline.text_encoder(uncond_text_input_ids)[0]

    with torch.no_grad():
        image = image_generation_pipeline(pipeline, ori_text_embeddings, text_embeddings, uncond_embeddings, 
                                num_inference_steps=num_inference_steps, 
                                guidance_scale_omega=guidance_scale_omega, 
                                guidance_scale_omega_a=guidance_scale_omega_a,
                                height=height, width=width, 
                                pos_text_embeddings=pos_text_embeddings,
                                neg_text_embeddings=neg_text_embeddings)

    return image

placeholder_token = args.placeholder_token
pos_token = args.pos_token
neg_token = args.neg_token
prompt_ori = args.prompt_ori
prompt = args.prompt

output_path = f"{args.model_path}/output"
SEED = args.seed
torch.manual_seed(SEED)
pipe = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16,safety_checker=None,
                                                requires_safety_checker=False,).to(device)

if args.iter is not None:
    for ii in range(args.iter):
        prompt_out = prompt.replace(" ", "-")
        if not os.path.exists(f"{output_path}/{SEED}_{str(ii)}_{args.weight}_{args.omega_a}_{args.omega}_{prompt_out}.png"):
            image = get_pipe_image(pipe, prompt_ori, prompt, placeholder_token, pos_token, neg_token, num_inference_steps=50, 
                                   guidance_scale_omega=args.omega, guidance_scale_omega_a=args.omega_a, height=512, width=512)

            if not os.path.exists(output_path):
                os.makedirs(output_path)
            image.save(f"{output_path}/{SEED}_{str(ii)}_{args.weight}_{args.omega_a}_{args.omega}_{prompt_out}.png")
else:
    prompt_out = prompt.replace(" ", "-")
    if not os.path.exists(f"{output_path}/{SEED}_{args.weight}_{args.omega_a}_{args.omega}_{prompt_out}.png"):
        image = get_pipe_image(pipe, prompt_ori, prompt, placeholder_token, pos_token, neg_token, num_inference_steps=50, 
                               guidance_scale_omega=args.omega, guidance_scale_omega_a=args.omega_a, height=512, width=512)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        image.save(f"{output_path}/{SEED}_{args.weight}_{args.omega_a}_{args.omega}_{prompt_out}.png")
