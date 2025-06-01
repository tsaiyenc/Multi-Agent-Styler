#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import os
import argparse
import logging
from os.path import join as ospj

import diffusers
import accelerate
import numpy as np
import transformers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision

import cv2
from PIL import Image
from tqdm import trange, tqdm
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
import matplotlib.pyplot as plt
from einops import rearrange
import torchvision.transforms.functional as TF

if is_wandb_available():
    import wandb

check_min_version("0.20.0")
logger = get_logger(__name__)


class DreamStylerDataset(torch.utils.data.Dataset):
    template = "A painting in the style of {}"

    def __init__(
        self,
        image_path,
        tokenizer,
        size=512,
        repeats=100,
        prob_flip=0.5,
        placeholder_tokens="*",
        center_crop=False,
        is_train=True,
        num_stages=1,
        context_prompt=None,
    ):
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_tokens = placeholder_tokens
        self.center_crop = center_crop
        self.prob_flip = prob_flip
        self.repeats = repeats if is_train else 1
        self.num_stages = num_stages

        if not isinstance(self.placeholder_tokens, list):
            self.placeholder_tokens = [self.placeholder_token]

        self.flip = torchvision.transforms.RandomHorizontalFlip(p=self.prob_flip)

        self.image_path = image_path
        self.prompt = self.template if context_prompt is None else context_prompt

    def __getitem__(self, index):
        image = Image.open(self.image_path).convert("RGB")
        image = np.array(image).astype(np.uint8)
        prompt = self.prompt

        tokens = []
        for t in range(self.num_stages):
            placeholder_string = self.placeholder_tokens[t]
            prompt_t = prompt.format(placeholder_string)

            tokens.append(
                self.tokenizer(
                    prompt_t,
                    padding="max_length",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids[0]
            )

        if self.center_crop:
            h, w = image.shape[0], image.shape[1]
            min_hw = min(h, w)
            image = image[
                (h - min_hw) // 2 : (h + min_hw) // 2,
                (w - min_hw) // 2 : (w + min_hw) // 2,
            ]

        image = Image.fromarray(image)
        image = image.resize((self.size, self.size), resample=Image.LANCZOS)
        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)

        return {
            "input_ids": tokens,
            "pixel_values": image,
        }

    def __len__(self):
        return self.repeats

def get_attn_hook(emb, ret, heads, scale):
    def hook(self, sin, sout):
        q = self.to_q(sin[0])
        k = self.to_k(emb)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=heads), (q, k))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * scale
        attn = sim.softmax(dim=-1)
        ret["attn"] = attn.detach()
    return hook


def get_attention_overlay(attn_map_flat, image_tensor):
    """
    Overlay the attention map on the original image and return the overlay image.
    attn_map_flat: 1D tensor, e.g. [64]
    image_tensor: [3, 224, 224]
    return: overlay (np.ndarray, shape [224, 224, 3])
    """
    numel = attn_map_flat.numel()
    side = int(np.sqrt(numel))
    if side * side != numel:
        print(f"Warning: Cannot reshape attention map of length {numel} to square")
        return None
    attn_map = attn_map_flat.view(side, side).numpy()
    attn_map = cv2.resize(attn_map, (224, 224), interpolation=cv2.INTER_CUBIC)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)
    heatmap = cv2.applyColorMap(np.uint8(attn_map * 255), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float32) / 255.0
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = (image - image.min()) / (image.max() - image.min() + 1e-6)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    overlay = 0.6 * image + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay


def visualize_attention_on_image(attn_b, image_tensor, input_ids, tokenizer, save_path):
    """
    Visualize the attention map for each token over the image and save as a grid image.
    attn_b: Tensor of shape [num_tokens, num_patches] (each token's attention to spatial patches; num_patches is usually 64 for 8x8, but may vary by model or resolution)
    image_tensor: Tensor of shape [3, 224, 224]
    input_ids: Tensor of shape [num_tokens]
    tokenizer: tokenizer instance
    save_path: str
    """
    num_tokens = attn_b.shape[0]
    max_per_row = 5
    ncols = min(num_tokens, max_per_row)
    nrows = (num_tokens + max_per_row - 1) // max_per_row
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    if nrows == 1:
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)
    for i in range(num_tokens):
        row, col = divmod(i, max_per_row)
        ax = axes[row, col]
        attn_map_flat = attn_b[i].detach().cpu()
        overlay = get_attention_overlay(attn_map_flat, image_tensor)
        if overlay is None:
            continue
        ax.imshow(overlay)
        ax.axis('off')
        token_str = tokenizer.decode([input_ids[i]], skip_special_tokens=True)
        ax.set_title(token_str, fontsize=8)
    # Hide unused axes
    for i in range(num_tokens, nrows * ncols):
        row, col = divmod(i, max_per_row)
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def visualize_placeholder_vs_negative_attention(
    attn_b, input_ids, tokenizer, placeholder_token_ids, negative_token_ids, image_tensor, save_path
):
    """
    Visualize and compare the attention maps of all placeholder tokens and negative prompt tokens over the image, saving as a single grid image.
    attn_b: [num_tokens, num_patches] (num_patches is usually 64 for 8x8, but may vary by model or resolution)
    input_ids: [num_tokens]
    tokenizer: tokenizer instance
    placeholder_token_ids: list of int
    negative_token_ids: list of int
    image_tensor: [3, 224, 224]
    save_path: str
    """
    input_ids_list = input_ids.tolist()
    placeholder_indices = [i for i, tid in enumerate(input_ids_list) if tid in placeholder_token_ids]
    negative_indices = [i for i, tid in enumerate(input_ids_list) if tid in negative_token_ids]


    def get_attn_images(indices, label):
        images = []
        titles = []
        for token_idx in indices:
            attn_map_flat = attn_b[token_idx].detach().cpu()
            overlay = get_attention_overlay(attn_map_flat, image_tensor)
            if overlay is None:
                continue
            images.append(overlay)
            token_str = tokenizer.decode([input_ids[token_idx]], skip_special_tokens=True)
            titles.append(f"{label}: {token_str}")
        return images, titles

    placeholder_imgs, placeholder_titles = get_attn_images(placeholder_indices, "Placeholder")
    negative_imgs, negative_titles = get_attn_images(negative_indices, "Negative")

    n_placeholder = len(placeholder_imgs)
    n_negative = len(negative_imgs)
    ncols = max(n_placeholder, n_negative)
    nrows = 2 if n_placeholder > 0 and n_negative > 0 else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    if nrows == 1:
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    # Placeholder row
    for i in range(ncols):
        ax = axes[0, i]
        if i < n_placeholder:
            ax.imshow(placeholder_imgs[i])
            ax.set_title(f"Placeholder: {placeholder_titles[i]}", fontsize=8)
        else:
            ax.axis('off')
        ax.axis('off')

    # Negative row
    if nrows == 2:
        for i in range(ncols):
            ax = axes[1, i]
            if i < n_negative:
                ax.imshow(negative_imgs[i])
                ax.set_title(f"Negative: {negative_titles[i]}", fontsize=8)
            else:
                ax.axis('off')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def train(opt):
    accelerator = init_accelerator_and_logger(logger, opt)
    (
        train_dataset,
        train_dataloader,
        placeholder_tokens,
        placeholder_token_ids,
        tokenizer,
        text_encoder,
        noise_scheduler,
        optimizer,
        lr_scheduler,
        vae,
        unet,
        weight_dtype,
    ) = init_model_and_dataset(accelerator, logger, opt)

    # process negative prompt
    negative_token_ids = []
    if hasattr(opt, "negative_prompt") and opt.negative_prompt is not None:
        if isinstance(opt.negative_prompt, str):
            negative_prompts = [s.strip() for s in opt.negative_prompt.split(",") if s.strip()]
            # negative_prompts = [opt.negative_prompt]
        else:
            negative_prompts = opt.negative_prompt
        for neg_prompt in negative_prompts:
            ids = tokenizer.encode(neg_prompt, add_special_tokens=False)
            negative_token_ids.extend(ids)
    negative_token_ids = list(set(negative_token_ids))

    # do we need this?
    if opt.resume_from_checkpoint:
        raise NotImplementedError

    # keep original embeddings as reference
    orig_embeds_params = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight.data.clone()
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Total optimization steps = {opt.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {opt.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {opt.gradient_accumulation_steps}")


    def register_attn_hook(attn_module, encoder_hidden_states, attn_hook_result):
        heads = attn_module.heads
        scale = attn_module.scale
        hook = get_attn_hook(encoder_hidden_states, attn_hook_result, heads, scale)
        handle = attn_module.register_forward_hook(hook)
        return handle, heads

    attn_modules = {
        'mid': unet.mid_block.attentions[0].transformer_blocks[0].attn2,
        'down1': unet.down_blocks[1].attentions[0].transformer_blocks[0].attn2,
    }
    attn_hook_results = {
        'mid': {},
        'down1': {},
    }

    os.makedirs(os.path.join(opt.output_dir, "attn_vis"), exist_ok=True)

    text_encoder.train()
    progress_bar = tqdm(range(opt.max_train_steps), disable=not accelerator.is_local_main_process)
    for step in progress_bar:
        try:
            batch = next(iters)
        except (UnboundLocalError, StopIteration, TypeError):
            iters = iter(train_dataloader)
            batch = next(iters)

        with accelerator.accumulate(text_encoder):
            # convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype))
            latents = latents.latent_dist.sample().detach() * vae.config.scaling_factor

            # sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Dreamstyler: get index in stage (T) axis
            max_timesteps = noise_scheduler.config.num_train_timesteps
            index_stage = (timesteps / max_timesteps * opt.num_stages).long()

            # add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # get the text embedding for conditioning
            # Dreamstyler: batch["input_ids"] is [T x bsz x 77]-dim shape
            # and if bsz > 1, timesteps have multiple arbitary t values
            # so that input_ids variable should be proprocessed
            # to be matched to appropriate timesteps
            input_ids = torch.empty_like(batch["input_ids"][0])
            for n in range(bsz):
                input_ids[n] = batch["input_ids"][index_stage[n]][n]
            encoder_hidden_states = text_encoder(input_ids)[0].to(weight_dtype)


            # Register hooks for both mid and down1 attention modules
            handles = {}
            heads_dict = {}
            for key in attn_modules:
                handles[key], heads_dict[key] = register_attn_hook(attn_modules[key], encoder_hidden_states, attn_hook_results[key])

            # predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Remove hooks
            for h in handles.values():
                h.remove()

            # get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            loss_main = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            

            attn = attn_hook_results['mid'].get("attn")  # mid block
            down1_attn = attn_hook_results['down1'].get("attn")  # down1 (down block) attn
            heads = heads_dict['mid']
            down1_heads = heads_dict['down1']

            def disentangle_loss(attn, heads, input_ids, placeholder_token_ids, negative_token_ids, bsz, loss_type):
                if attn is None or not negative_token_ids:
                    return 0.0
                attn = attn.view(bsz, heads, attn.shape[-2], attn.shape[-1])
                placeholder_token_id = placeholder_token_ids[0]
                placeholder_mask = (input_ids == placeholder_token_id)  # [B, T]
                loss = 0.0
                count = 0
                for b in range(bsz):
                    mask = placeholder_mask[b]  # [T]
                    if mask.sum() == 0:
                        continue
                    attn_b = attn[b].mean(0)  # [Tq, Tk]
                    placeholder_indices = mask.nonzero(as_tuple=True)[0]
                    negative_mask = torch.zeros_like(mask, dtype=torch.bool)
                    for neg_id in negative_token_ids:
                        negative_mask |= (input_ids[b] == neg_id)
                    negative_indices = negative_mask.nonzero(as_tuple=True)[0]
                    if len(negative_indices) == 0:
                        continue
                    for pi in placeholder_indices:
                        attn_placeholder = attn_b[:, pi]
                        if loss_type == "cosine":
                            for ni in negative_indices:
                                attn_negative = attn_b[:, ni]
                                sim = F.cosine_similarity(attn_placeholder, attn_negative, dim=0)
                                loss += sim
                                count += 1
                        elif loss_type in ["kl", "js"]:
                            for ni in negative_indices:
                                attn_negative = attn_b[:, ni]
                                attn_placeholder_ = attn_placeholder + 1e-8
                                attn_negative_ = attn_negative + 1e-8
                                attn_placeholder_ = attn_placeholder_ / attn_placeholder_.sum()
                                attn_negative_ = attn_negative_ / attn_negative_.sum()
                                if loss_type == "kl":
                                    kl = torch.sum(attn_placeholder_ * torch.log(attn_placeholder_ / attn_negative_))
                                    loss += -kl
                                elif loss_type == "js":
                                    m = 0.5 * (attn_placeholder_ + attn_negative_)
                                    m = m + 1e-8
                                    kl_pm = torch.sum(attn_placeholder_ * torch.log(attn_placeholder_ / m))
                                    kl_qm = torch.sum(attn_negative_ * torch.log(attn_negative_ / m))
                                    js = 0.5 * (kl_pm + kl_qm)
                                    loss += -js
                                count += 1
                if count > 0:
                    loss /= count
                else:
                    loss = torch.tensor(0.0, device=attn.device)
                return loss
            
            disentangle_loss_mid, disentangle_loss_down1 = 0.0, 0.0
            if (opt.disentangle_loss_weight_mid != 0.0):
                disentangle_loss_mid = disentangle_loss(attn, heads, input_ids, placeholder_token_ids, negative_token_ids, bsz, opt.attn_loss)
            if (opt.disentangle_loss_weight_down1 != 0.0):
                disentangle_loss_down1 = disentangle_loss(down1_attn, down1_heads, input_ids, placeholder_token_ids, negative_token_ids, bsz, opt.attn_loss)
            loss = loss_main + opt.disentangle_loss_weight_mid * disentangle_loss_mid + opt.disentangle_loss_weight_down1 * disentangle_loss_down1

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # let's make sure we don't update any embedding weights
            # besides the newly added token
            index_no_updates = ~torch.isin(
                torch.arange(len(tokenizer)),
                torch.tensor(placeholder_token_ids),
            )
            with torch.no_grad():
                emb1 = accelerator.unwrap_model(text_encoder).get_input_embeddings()
                emb2 = orig_embeds_params[index_no_updates]
                emb1.weight[index_no_updates] = emb2

            if accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    mid=disentangle_loss_mid*opt.disentangle_loss_weight_mid if attn is not None and negative_token_ids else 0.0,
                    down1=disentangle_loss_down1*opt.disentangle_loss_weight_down1 if down1_attn is not None and negative_token_ids else 0.0,
                    main=loss_main.item(),
                )

            if step % 100 == 0 and attn is not None and accelerator.is_main_process:
                os.makedirs(os.path.join(opt.output_dir, "attn_vis", f"step_{step}"), exist_ok=True)

                attn = attn.view(bsz, heads, attn.shape[-2], attn.shape[-1])  # [B, H, 64, 77]
                for b in range(bsz):
                    if opt.visualize_mid_attn:
                        attn_b = attn[b].mean(0)  # [64, 77]
                        attn_b = attn_b.transpose(0, 1)  # [77, 64]
                        save_path = os.path.join(opt.output_dir, "attn_vis", f"step_{step}", f"all_tokens_b{b}.png")
                        visualize_attention_on_image(
                            attn_b,
                            batch["pixel_values"][b],
                            input_ids[b],
                            tokenizer,
                            save_path
                        )
                        save_path2 = os.path.join(opt.output_dir, "attn_vis", f"step_{step}", f"ph_vs_neg_b{b}.png")
                        visualize_placeholder_vs_negative_attention(
                            attn_b,
                            input_ids[b],
                            tokenizer,
                            placeholder_token_ids,
                            negative_token_ids,
                            batch["pixel_values"][b],
                            save_path2
                        )
                    if opt.visualize_down1_attn and down1_attn is not None:
                        down1_attn = down1_attn.view(bsz, down1_heads, down1_attn.shape[-2], down1_attn.shape[-1])  # [B, H, 64, 77]
                        attn_b = down1_attn[b].mean(0)  # [64, 77]
                        attn_b = attn_b.transpose(0, 1)  # [77, 64]
                        save_path_down1 = os.path.join(opt.output_dir, "attn_vis", f"step_{step}", f"ph_vs_neg_b{b}_down1.png")
                        visualize_placeholder_vs_negative_attention(
                            attn_b,
                            input_ids[b],
                            tokenizer,
                            placeholder_token_ids,
                            negative_token_ids,
                            batch["pixel_values"][b],
                            save_path_down1
                        )

        if accelerator.sync_gradients:
            if accelerator.is_main_process and (step + 1) % opt.save_steps == 0:
                save(accelerator, text_encoder, placeholder_tokens, placeholder_token_ids, step + 1, opt)

    accelerator.wait_for_everyone()
    save(accelerator, text_encoder, placeholder_tokens, placeholder_token_ids, "final", opt)
    accelerator.end_training()


def init_accelerator_and_logger(logger, opt):
    logging_dir = ospj(opt.output_dir, opt.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=opt.output_dir,
        logging_dir=logging_dir,
    )
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        mixed_precision=opt.mixed_precision,
        log_with=opt.report_to,
        project_config=accelerator_project_config,
    )

    if opt.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it"
                " for logging during training."
            )

    # make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # handle the repository creation
    if accelerator.is_main_process:
        if opt.output_dir is not None:
            os.makedirs(opt.output_dir, exist_ok=True)

    os.makedirs(ospj(opt.output_dir, "embedding"), exist_ok=True)
    return accelerator


def init_model_and_dataset(accelerator, logger, opt, without_dataset=False):
    if opt.seed is not None:
        set_seed(opt.seed)

    if opt.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(opt.tokenizer_name)
    elif opt.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            opt.pretrained_model_name_or_path,
            subfolder="tokenizer",
        )

    noise_scheduler = DDPMScheduler.from_pretrained(
        opt.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        opt.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=opt.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        opt.pretrained_model_name_or_path,
        subfolder="vae",
        revision=opt.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        opt.pretrained_model_name_or_path,
        subfolder="unet",
        revision=opt.revision,
    )

    # DreamStyler: TODO: support multi-vector TI
    if opt.num_vectors > 1:
        raise NotImplementedError

    # DreamStyler: add new textual embeddings
    placeholder_tokens = [
        f"{opt.placeholder_token}-T{t}" for t in range(opt.num_stages)
    ]
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {opt.placeholder_token}."
            " Please pass a different `placeholder_token` that is not already in the tokenizer."
        )

    # convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(opt.initializer_token, add_special_tokens=False)
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # initialize the newly added placeholder token
    # with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    # freeze vae and unet and text encoder (except for the token embeddings)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    if opt.gradient_checkpointing:
        # keep unet in train mode if we are using gradient checkpointing to save memory.
        # the dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if opt.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import version
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs."
                    " If you observe problems during training, please update xFormers"
                    " to at least 0.0.17."
                    " See https://huggingface.co/docs/diffusers/main/en/optimization/xformers"
                    " for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if opt.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if opt.scale_lr:
        opt.learning_rate = (
            opt.learning_rate
            * opt.gradient_accumulation_steps
            * opt.train_batch_size
            * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),
        lr=opt.learning_rate,
        betas=(opt.adam_beta1, opt.adam_beta2),
        weight_decay=opt.adam_weight_decay,
        eps=opt.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        opt.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=opt.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=opt.max_train_steps * accelerator.num_processes,
        num_cycles=opt.lr_num_cycles,
    )

    if without_dataset:
        train_dataset, train_dataloader = None, None
        text_encoder, optimizer, lr_scheduler = accelerator.prepare(
            text_encoder,
            optimizer,
            lr_scheduler,
        )
    else:
        train_dataset = DreamStylerDataset(
            image_path=opt.train_image_path,
            tokenizer=tokenizer,
            size=opt.resolution,
            placeholder_tokens=placeholder_tokens,
            repeats=opt.max_train_steps,
            center_crop=opt.center_crop,
            is_train=True,
            context_prompt=opt.context_prompt,
            num_stages=opt.num_stages,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.train_batch_size,
            shuffle=True,
            num_workers=opt.dataloader_num_workers,
        )
        text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )

    text_encoder, optimizer, lr_scheduler = accelerator.prepare(
        text_encoder,
        optimizer,
        lr_scheduler,
    )

    # for mixed precision training we cast all non-trainable weigths
    # (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference,
    # keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # we need to initialize the trackers we use, and also store our configuration.
    # the trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreamstyler", config=vars(opt))

    return (
        train_dataset,
        train_dataloader,
        placeholder_tokens,
        placeholder_token_ids,
        tokenizer,
        text_encoder,
        noise_scheduler,
        optimizer,
        lr_scheduler,
        vae,
        unet,
        weight_dtype,
    )


def save(
    accelerator,
    text_encoder,
    placeholder_tokens,
    placeholder_token_ids,
    prefix,
    opt,
):
    prefix = f"{prefix:04d}" if isinstance(prefix, int) else prefix

    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings()
    embeds_dict = {}
    for token, token_id in zip(placeholder_tokens, placeholder_token_ids):
        embeds_dict[token] = learned_embeds.weight[token_id].detach().cpu()
    torch.save(embeds_dict, ospj(opt.output_dir, "embedding", f"{prefix}.bin"))


def get_options():
    parser = argparse.ArgumentParser()

    # DreamStyler arguments
    parser.add_argument(
        "--context_prompt",
        type=str,
        default=None,
        help="Additional context prompt to enhance training performance.",
    )
    parser.add_argument(
        "--num_stages",
        type=int,
        default=6,
        help="The number of the stages (denoted as T) used in multi-stage TI.",
    )

    # original textual inversion arguments
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        # default=None,
        # required=True,
        default="runwayml/stable-diffusion-v1-5",
        # default="stabilityai/stable-diffusion-2-1",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_image_path",
        type=str,
        default=None,
        required=True,
        help="A path of training image.",
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token",
        type=str,
        default="painting",
        help="A token to use as initializer word.",
    )
    parser.add_argument(
        "--learnable_property",
        type=str,
        default="style",
        help="Choose between 'object' and 'style'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dreamstyler",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the"
            " train/validation dataset will be resized to this resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        # default=8,
        default=12,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=500,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Whether or not to use gradient checkpointing"
            " to save memory at the expense of slower backward pass."
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        # default=1e-4,
        default=2e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help=(
            "Scale the learning rate by the number of GPUs,"
            " gradient accumulation steps, and batch size."
        ),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            "The scheduler type to use. Choose between"
            " ['linear', 'cosine', 'cosine_with_restarts', 'polynomial',"
            " 'constant', 'constant_with_warmup']"
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        # default=500,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=(
            "Number of subprocesses to use for data loading."
            " 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory."
            " Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs."
            " Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            "The integration to report the results and logs to."
            ' Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`.'
            ' Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=5,
        help=(
            "Number of images that should be generated"
            " during validation with `validation_prompt`.",
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates."
            " These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint."
            " Use a path saved by `--checkpointing_steps`,"
            ' or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help=(
            "If specified save the checkpoint not in `safetensors` format,"
            " but in original PyTorch format instead.",
        ),
    )
    
    parser.add_argument(
        "--negative_prompt",    
        type=str,
        default=None,
        help=(
            "A negative prompt that is used to guide the model away from certain concepts."
            " This can be useful to prevent the model from learning unwanted features."
        ),
    )
    
    parser.add_argument(
        "--disentangle_loss_weight_mid",
        type=float,
        default=0.1,
        help=(
            "Weight for the disentangle loss, which encourages the model to learn"
            " separate representations for different tokens."
        ),
    )
    
    parser.add_argument(
        "--disentangle_loss_weight_down1",
        type=float,
        default=0.0,
        help=(
            "Weight for the disentangle loss, which encourages the model to learn"
            " separate representations for different tokens."
        ),
    )

    parser.add_argument(
        "--visualize_mid_attn",
        action="store_true",
        help="Whether to visualize mid attention maps during training.",
    )
    parser.add_argument(
        "--visualize_down1_attn",
        action="store_true",
        help="Whether to visualize down1 attention maps during training.",
    )
    
    parser.add_argument(
        "--attn_loss",
        type=str,
        default="cosine",
        choices=["cosine", "kl", "js"],
        help=(
            "The type of attention loss to use. Choose between"
            " ['cosine', 'kl', 'js']."
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank not in (-1, args.local_rank):
        args.local_rank = env_local_rank

    return args


if __name__ == "__main__":
    opt = get_options()
    train(opt)