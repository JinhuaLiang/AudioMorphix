import os
import copy
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# from basicsr.utils import img2tensor
from diffusers import DDIMScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from transformers import (
    ClapTextModelWithProjection,
    RobertaTokenizer,
    RobertaTokenizerFast,
    SpeechT5HifiGan,
)
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from typing import List, Optional, Union

from src.utils.factory import (
    normalize_along_channel,
    project_onto_tangent_space,
    identity_projection,
)


log = logging.getLogger(__name__)


class Sampler(DiffusionPipeline):
    r"""Core of Audio Morphix that samples latent with inversed latent and cross-attend trajactory."""

    def __init__(
        self,
        vae: AutoencoderKL,
        tokenizer: Union[RobertaTokenizer, RobertaTokenizerFast],
        text_encoder: ClapTextModelWithProjection,
        unet: UNet2DConditionModel,
        feature_estimator: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        device: torch.device = torch.device("cpu"),
        precision: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            feature_estimator=feature_estimator,
            scheduler=scheduler,
        )
        self = self.to(torch_device=device, torch_dtype=precision)
        self._device = device

    def edit(
        self,
        prompt: str,
        mode,
        edit_kwargs,
        prompt_replace: str = None,
        negative_prompt: str = None,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        latent: Optional[torch.FloatTensor] = None,
        start_time: int = 50,
        energy_scale: float = 0,
        SDE_strength: float = 0.4,
        SDE_strength_un: float = 0,
        latent_noise_ref: Optional[torch.FloatTensor] = None,
        bg_to_fg_ratio: float = 0.5,
        disable_tangent_proj: bool = False,
        alg: str = "D+",  
    ):
        log.info("Start Editing:")
        self.alg = alg

        # Select projection function
        if disable_tangent_proj:
            log.info("Use guidance directly.")
            self.proj_fn = identity_projection
        else:
            log.info("Project guidance onto tangent space.")
            self.proj_fn = project_onto_tangent_space
            
        # Generate source text embedding
        text_input = self._encode_text(prompt)
        if prompt_replace is not None:
            text_replace = self._encode_text(prompt_replace)
        else:
            text_replace = text_input

        # Generate null text embedding for CFG
        prompt_uncond = "" if negative_prompt is None else negative_prompt
        text_uncond = self._encode_text(prompt_uncond)

        # Text condition for the current trajectory
        context = self._stack_text(text_uncond, text_input)

        self.scheduler.set_timesteps(num_inference_steps)
        dict_mask = edit_kwargs["dict_mask"] if "dict_mask" in edit_kwargs else None
        time_scale = num_inference_steps / 50 # scale the editing operation (default is 50)

        for i, t in enumerate(tqdm(self.scheduler.timesteps[-start_time:])):
            next_timestep = min(
                t
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps,
                999,
            )
            next_timestep = max(next_timestep, 0)
            if energy_scale == 0 or alg == "D":
                repeat = 1
            elif int(20*time_scale) < i < int(30*time_scale) and i % 2 == 0:
                repeat = 3
            else:
                repeat = 1

            for ri in range(repeat):
                latent_in = torch.cat([latent.unsqueeze(2)] * 2)
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_in,
                        t,
                        encoder_hidden_states=context["embed"] if self.use_cross_attn else None,
                        class_labels=None if self.use_cross_attn else context["embed"],
                        encoder_attention_mask=context["mask"] if self.use_cross_attn else None,
                        mask=dict_mask,
                        save_kv=False,
                        mode=mode,
                        iter_cur=i,
                    )["sample"].squeeze(2)

                noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_prediction_text - noise_pred_uncond
                )

                if (
                    energy_scale != 0
                    and int(i*time_scale) < 30
                    and (alg == "D" or i % 2 == 0 or i < int(10*time_scale))
                ):
                    # editing guidance
                    noise_pred_org = noise_pred
                    if mode == "move":
                        guidance = self.guidance_move(
                            latent=latent,
                            latent_noise_ref=latent_noise_ref[-(i + 1)],
                            t=t,
                            context=text_input,
                            context_base=text_input,
                            energy_scale=energy_scale,
                            **edit_kwargs
                        )
                    elif mode == "drag":
                        guidance = self.guidance_drag(
                            latent=latent,
                            latent_noise_ref=latent_noise_ref[-(i + 1)],
                            t=t,
                            context=text_input,
                            context_base=text_input,
                            energy_scale=energy_scale,
                            **edit_kwargs
                        )
                    elif mode == "landmark":
                        guidance = self.guidance_landmark(
                            latent=latent,
                            latent_noise_ref=latent_noise_ref[-(i + 1)],
                            t=t,
                            context=text_input,
                            context_base=text_input,
                            energy_scale=energy_scale,
                            **edit_kwargs
                        )
                    elif mode == "appearance":
                        guidance = self.guidance_appearance(
                            latent=latent,
                            latent_noise_ref=latent_noise_ref[-(i + 1)],
                            t=t,
                            context=text_input,
                            context_base=text_input,
                            energy_scale=energy_scale,
                            **edit_kwargs
                        )
                    elif mode == "paste":
                        guidance = self.guidance_paste(
                            latent=latent,
                            latent_noise_ref=latent_noise_ref[-(i + 1)],
                            t=t,
                            context=text_input,
                            context_base=text_input,
                            context_replace=text_replace,
                            energy_scale=energy_scale,
                            **edit_kwargs
                        )
                    elif mode == "mix":
                        guidance = self.guidance_mix(
                            latent=latent,
                            latent_noise_ref=latent_noise_ref[-(i + 1)],
                            t=t,
                            context=text_input,
                            context_base=text_input,
                            context_replace=text_replace,
                            energy_scale=energy_scale,
                            **edit_kwargs
                        )
                    elif mode == "remove":
                        guidance = self.guidance_remove(
                            latent=latent,
                            latent_noise_ref=latent_noise_ref[-(i + 1)],
                            t=t,
                            context=text_input,
                            context_base=text_input,
                            context_replace=text_replace,
                            energy_scale=energy_scale,
                            **edit_kwargs
                        )
                    elif mode == "style_transfer":
                        guidance = self.guidance_style_transfer(
                            latent=latent,
                            latent_noise_ref=latent_noise_ref[-(i + 1)],
                            t=t,
                            context=text_input,
                            context_base=text_input,
                            energy_scale=energy_scale,
                            **edit_kwargs
                        )
                    # Project guidance onto z_t
                    guidance = self.proj_fn(guidance, latent)
                    noise_pred = noise_pred + guidance # NOTE: weighted sum?
                else:
                    noise_pred_org = None

                # zt->zt-1
                prev_timestep = (
                    t
                    - self.scheduler.config.num_train_timesteps
                    // self.scheduler.num_inference_steps
                )
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[prev_timestep]
                    if prev_timestep >= 0
                    else self.scheduler.final_alpha_cumprod
                )
                beta_prod_t = 1 - alpha_prod_t

                if self.scheduler.config.prediction_type == "epsilon":
                    pred_original_sample = (
                        latent - beta_prod_t ** (0.5) * noise_pred
                    ) / alpha_prod_t ** (0.5)
                    pred_epsilon = noise_pred
                    pred_epsilon_org = noise_pred_org

                elif self.scheduler.config.prediction_type == "v_prediction":
                    pred_original_sample = (alpha_prod_t**0.5) * latent - (beta_prod_t**0.5) * noise_pred
                    pred_epsilon = (alpha_prod_t**0.5) * noise_pred + (beta_prod_t**0.5) * latent
                    if noise_pred_org is not None:
                        pred_epsilon_org = (alpha_prod_t**0.5) * noise_pred_org + (beta_prod_t**0.5) * latent
                    else:
                        pred_epsilon_org = None

                if int(10*time_scale) < i < int(20*time_scale):
                    eta, eta_rd = SDE_strength_un, SDE_strength
                else:
                    eta, eta_rd = 0.0, 0.0

                variance = self.scheduler._get_variance(t, prev_timestep)
                std_dev_t = eta * variance ** (0.5)
                std_dev_t_rd = eta_rd * variance ** (0.5)
                if noise_pred_org is not None:
                    pred_sample_direction_rd = (
                        1 - alpha_prod_t_prev - std_dev_t_rd**2
                    ) ** (0.5) * pred_epsilon_org
                    pred_sample_direction = (
                        1 - alpha_prod_t_prev - std_dev_t**2
                    ) ** (0.5) * pred_epsilon_org
                else:
                    pred_sample_direction_rd = (
                        1 - alpha_prod_t_prev - std_dev_t_rd**2
                    ) ** (0.5) * pred_epsilon
                    pred_sample_direction = (
                        1 - alpha_prod_t_prev - std_dev_t**2
                    ) ** (0.5) * pred_epsilon

                latent_prev = (
                    alpha_prod_t_prev ** (0.5) * pred_original_sample
                    + pred_sample_direction
                )
                latent_prev_rd = (
                    alpha_prod_t_prev ** (0.5) * pred_original_sample
                    + pred_sample_direction_rd
                )

                # Regional SDE
                if (eta_rd > 0 or eta > 0) and alg == "D+":
                    variance_noise = torch.randn_like(latent_prev)
                    variance_rd = std_dev_t_rd * variance_noise
                    variance = std_dev_t * variance_noise

                    if mode == "move":
                        mask = (
                            F.interpolate(
                                edit_kwargs["mask_x0"][None, None],
                                (
                                    edit_kwargs["mask_cur"].shape[-2],
                                    edit_kwargs["mask_cur"].shape[-1],
                                ),
                            )
                            > 0.5
                        ).float()
                        mask = ((edit_kwargs["mask_cur"] + mask) > 0.5).float()
                        mask = (
                            F.interpolate(
                                mask, (latent_prev.shape[-2], latent_prev.shape[-1])
                            )
                            > 0.5
                        ).to(dtype=latent.dtype)
                    elif mode == "drag":
                        mask = F.interpolate(
                            edit_kwargs["mask_x0"][None, None],
                            (latent_prev[-1].shape[-2], latent_prev[-1].shape[-1]),
                        )
                        mask = (mask > 0).to(dtype=latent.dtype)
                    elif mode == "landmark":
                        mask = torch.ones_like(latent_prev)
                    elif (
                        mode == "appearance"
                        or mode == "paste"
                        or mode == "remove"
                        or mode == "mix"
                    ):
                        mask = F.interpolate(
                            edit_kwargs["mask_base_cur"].float(),
                            (latent_prev[-1].shape[-2], latent_prev[-1].shape[-1]),
                        )
                        mask = (mask > 0).to(dtype=latent.dtype)

                    latent_prev = (latent_prev + variance) * (1 - mask) + (
                        latent_prev_rd + variance_rd
                    ) * mask

                if repeat > 1:
                    with torch.no_grad():
                        alpha_prod_t = self.scheduler.alphas_cumprod[next_timestep]
                        alpha_prod_t_next = self.scheduler.alphas_cumprod[t]
                        beta_prod_t = 1 - alpha_prod_t

                        model_output = self.unet(
                            latent_prev.unsqueeze(2),
                            next_timestep,
                            class_labels=None if self.use_cross_attn else text_input["embed"],
                            encoder_hidden_states=text_input["embed"] if self.use_cross_attn else None,
                            encoder_attention_mask=text_input["mask"] if self.use_cross_attn else None,
                            mask=dict_mask,
                            save_kv=False,
                            mode=mode,
                            iter_cur=-2,
                        )["sample"].squeeze(2)

                        # Different scheduling options
                        if self.scheduler.config.prediction_type == "epsilon":
                            next_original_sample = (
                                latent_prev - beta_prod_t**0.5 * model_output
                            ) / alpha_prod_t**0.5
                            pred_epsilon = model_output
                        elif self.scheduler.config.prediction_type == "v_prediction":
                            next_original_sample = (alpha_prod_t**0.5) * latent_prev - (beta_prod_t**0.5) * model_output
                            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * latent_prev
                        
                        next_sample_direction = (
                            1 - alpha_prod_t_next
                        ) ** 0.5 * pred_epsilon
                        latent = (
                            alpha_prod_t_next**0.5 * next_original_sample
                            + next_sample_direction
                        )

            latent = latent_prev

        return latent

    def guidance_move(
        self,
        mask_x0,
        mask_x0_ref,
        mask_x0_keep,
        mask_tar,
        mask_cur,
        mask_keep,
        mask_other,
        mask_overlap,
        mask_non_overlap,
        latent,
        latent_noise_ref,
        t,
        up_ft_index,
        context,
        context_base,
        up_scale,
        resize_scale_x,
        resize_scale_y,
        energy_scale,
        w_edit,
        w_content,
        w_contrast,
        w_inpaint,
    ):
        cos = nn.CosineSimilarity(dim=1)
        loss_scale = [0.5, 0.5]
        with torch.no_grad():
            up_ft_tar = self.feature_estimator(
                sample=latent_noise_ref.squeeze(2),
                timestep=t,
                up_ft_indices=up_ft_index,
                class_labels=None if self.use_cross_attn else context_base["embed"],
                encoder_hidden_states=(
                    context_base["embed"] if self.use_cross_attn else None
                ),
                encoder_attention_mask=context_base["mask"] if self.use_cross_attn else None,
            )["up_ft"]
            up_ft_tar_org = copy.deepcopy(up_ft_tar)
            for f_id in range(len(up_ft_tar_org)):
                up_ft_tar_org[f_id] = F.interpolate(
                    up_ft_tar_org[f_id],
                    (
                        up_ft_tar_org[-1].shape[-2] * up_scale,
                        up_ft_tar_org[-1].shape[-1] * up_scale,
                    ),
                )

        latent = latent.detach().requires_grad_(True)
        for f_id in range(len(up_ft_tar)):
            up_ft_tar[f_id] = F.interpolate(
                up_ft_tar[f_id],
                (
                    int(up_ft_tar[-1].shape[-2] * resize_scale_y * up_scale),
                    int(up_ft_tar[-1].shape[-1] * resize_scale_x * up_scale),
                ),
            )

        up_ft_cur = self.feature_estimator(
            sample=latent,
            timestep=t,
            up_ft_indices=up_ft_index,
            class_labels=None if self.use_cross_attn else context["embed"],
            encoder_hidden_states=(
                context["embed"] if self.use_cross_attn else None
            ),
            encoder_attention_mask=context["mask"] if self.use_cross_attn else None,
        )["up_ft"]
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(
                up_ft_cur[f_id],
                (
                    up_ft_cur[-1].shape[-2] * up_scale,
                    up_ft_cur[-1].shape[-1] * up_scale,
                ),
            )
        # editing energy
        loss_edit = 0
        for f_id in range(len(up_ft_tar)):
            up_ft_cur_vec = (
                up_ft_cur[f_id][mask_cur.repeat(1, up_ft_cur[f_id].shape[1], 1, 1)]
                .view(up_ft_cur[f_id].shape[1], -1)
                .permute(1, 0)
            )  # (x1, D)
            up_ft_tar_vec = (
                up_ft_tar[f_id][mask_tar.repeat(1, up_ft_tar[f_id].shape[1], 1, 1)]
                .view(up_ft_tar[f_id].shape[1], -1)
                .permute(1, 0)
            )  # (x2, D)
            # Compute consine sim between `up_ft_cur_vec` and `up_ft_tar_vec`
            # up_ft_cur_vec_norm = up_ft_cur_vec / (up_ft_cur_vec.norm(dim=1, keepdim=True) + 1e-8)
            # up_ft_tar_vec_norm = up_ft_tar_vec / (up_ft_tar_vec.norm(dim=1, keepdim=True) + 1e-8)
            # sim = torch.mm(up_ft_cur_vec_norm, up_ft_tar_vec_norm.T) 

            sim = cos(up_ft_cur_vec, up_ft_tar_vec)
            # sim_global = cos(
            #     up_ft_cur_vec.mean(0, keepdim=True), up_ft_tar_vec.mean(0, keepdim=True)
            # )
            loss_edit = loss_edit + (w_edit / (1 + 4 * sim.mean())) * loss_scale[f_id]

        # Content energy
        loss_con = 0
        for f_id in range(len(up_ft_tar_org)):
            sim_other = cos(up_ft_tar_org[f_id], up_ft_cur[f_id])[0][mask_other[0, 0]]
            loss_con = (
                loss_con + w_content / (1 + 4 * sim_other.mean()) * loss_scale[f_id]
            )

        if mask_x0_ref is not None:
            mask_x0_ref_cur = (
                F.interpolate(
                    mask_x0_ref[None, None],
                    (mask_other.shape[-2], mask_other.shape[-1]),
                )
                > 0.5
            )
        else:
            mask_x0_ref_cur = mask_other

        for f_id in range(len(up_ft_tar)):
            # # Global
            # up_ft_cur_non_overlap_contrast = (
            #     up_ft_cur[f_id][
            #         mask_non_overlap.repeat(1, up_ft_cur[f_id].shape[1], 1, 1)
            #     ]
            #     .view(up_ft_cur[f_id].shape[1], -1)
            #     .permute(1, 0)
            # )
            # up_ft_tar_non_overlap_contrast = (
            #     up_ft_tar_org[f_id][
            #         mask_non_overlap.repeat(1, up_ft_tar_org[f_id].shape[1], 1, 1)
            #     ]
            #     .view(up_ft_tar_org[f_id].shape[1], -1)
            #     .permute(1, 0)
            # )
            # F sim
            up_ft_cur_non_overlap_sum = torch.sum(
                up_ft_cur[f_id] * mask_non_overlap.repeat(1, up_ft_cur[f_id].shape[1], 1, 1),
                dim=-2,
            )
            up_ft_tar_non_overlap_sum = torch.sum(
                up_ft_tar_org[f_id]
                * mask_non_overlap.repeat(1, up_ft_tar_org[f_id].shape[1], 1, 1),
                dim=-2,
            )  # feature for reference audio

            mask_sum = torch.sum(mask_non_overlap, dim=-2)

            up_ft_cur_non_overlap_contrast = (
                (up_ft_cur_non_overlap_sum / (mask_sum + 1e-8))
                .view(up_ft_cur[f_id].shape[1], -1)
                .permute(1, 0)
            )  # avoid dividing zero
            up_ft_tar_non_overlap_contrast = (
                (up_ft_tar_non_overlap_sum / (mask_sum + 1e-8))
                .view(up_ft_tar_org[f_id].shape[1], -1)
                .permute(1, 0)
            )
            
            sim_non_overlap_contrast = (
                cos(up_ft_cur_non_overlap_contrast, up_ft_tar_non_overlap_contrast) + 1.0
            ) / 2.0
            loss_con = loss_con + w_contrast * sim_non_overlap_contrast.mean() * loss_scale[f_id]
            
            up_ft_cur_non_overlap_inpaint = (
                up_ft_cur[f_id][
                    mask_non_overlap.repeat(1, up_ft_cur[f_id].shape[1], 1, 1)
                ]
                .view(up_ft_cur[f_id].shape[1], -1)
                .permute(1, 0)
                .mean(0, keepdim=True)
            )
            up_ft_tar_non_overlap_inpaint = (
                up_ft_tar_org[f_id][
                    mask_x0_ref_cur.repeat(1, up_ft_tar_org[f_id].shape[1], 1, 1)
                ]
                .view(up_ft_tar_org[f_id].shape[1], -1)
                .permute(1, 0)
                .mean(0, keepdim=True)
            )
            sim_inpaint = (
                cos(up_ft_cur_non_overlap_inpaint, up_ft_tar_non_overlap_inpaint) + 1.0
            ) / 2.0
            loss_con = loss_con + w_inpaint / (1 + 4 * sim_inpaint.mean())

        cond_grad_edit = torch.autograd.grad(
            loss_edit * energy_scale, latent, retain_graph=True
        )[0]
        cond_grad_con = torch.autograd.grad(loss_con * energy_scale, latent)[0]

        mask_edit1 = (mask_cur > 0.5).float()
        mask_edit1 = (
            F.interpolate(mask_edit1, (latent.shape[-2], latent.shape[-1])) > 0
        ).to(dtype=latent.dtype)
        # mask_edit2 = ((mask_keep + mask_non_overlap.float()) > 0.5).float()
        # mask_edit2 = (
        #     F.interpolate(mask_edit2, (latent.shape[-2], latent.shape[-1])) > 0
        # ).to(dtype=latent.dtype)
        mask_edit2 = 1-mask_edit1
        guidance = (
            cond_grad_edit.detach() * 8e-2 * mask_edit1
            + cond_grad_con.detach() * 8e-2 * mask_edit2
        )
        self.feature_estimator.zero_grad()

        return guidance

    def guidance_drag(
        self,
        mask_x0,
        mask_cur,
        mask_tar,
        mask_other,
        latent,
        latent_noise_ref,
        t,
        up_ft_index,
        up_scale,
        context,
        context_base,
        energy_scale,
        w_edit,
        w_inpaint,
        w_content,
        dict_mask=None,
    ):
        cos = nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            up_ft_tar = self.feature_estimator(
                sample=latent_noise_ref.squeeze(2),
                timestep=t,
                up_ft_indices=up_ft_index,
                class_labels=None if self.use_cross_attn else context_base["embed"],
                encoder_hidden_states=(
                    context_base["embed"] if self.use_cross_attn else None
                ),
                encoder_attention_mask=context_base["mask"] if self.use_cross_attn else None,
            )["up_ft"]
            for f_id in range(len(up_ft_tar)):
                up_ft_tar[f_id] = F.interpolate(
                    up_ft_tar[f_id],
                    (
                        up_ft_tar[-1].shape[-2] * up_scale,
                        up_ft_tar[-1].shape[-1] * up_scale,
                    ),
                )

        latent = latent.detach().requires_grad_(True)
        up_ft_cur = self.feature_estimator(
            sample=latent,
            timestep=t,
            up_ft_indices=up_ft_index,
            class_labels=None if self.use_cross_attn else context["embed"],
            encoder_hidden_states=(
                context["embed"] if self.use_cross_attn else None
            ),
            encoder_attention_mask=context["mask"] if self.use_cross_attn else None,
        )["up_ft"]
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(
                up_ft_cur[f_id],
                (
                    up_ft_cur[-1].shape[-2] * up_scale,
                    up_ft_cur[-1].shape[-1] * up_scale,
                ),
            )

        # moving loss
        loss_edit = 0
        for f_id in range(len(up_ft_tar)):
            for mask_cur_i, mask_tar_i in zip(mask_cur, mask_tar):
                up_ft_cur_vec = (
                    up_ft_cur[f_id][
                        mask_cur_i.repeat(1, up_ft_cur[f_id].shape[1], 1, 1)
                    ]
                    .view(up_ft_cur[f_id].shape[1], -1)
                    .permute(1, 0)
                )
                up_ft_tar_vec = (
                    up_ft_tar[f_id][
                        mask_tar_i.repeat(1, up_ft_tar[f_id].shape[1], 1, 1)
                    ]
                    .view(up_ft_tar[f_id].shape[1], -1)
                    .permute(1, 0)
                )
                sim = (cos(up_ft_cur_vec, up_ft_tar_vec) + 1.0) / 2.0
                loss_edit = loss_edit + w_edit / (1 + 4 * sim.mean())

                mask_overlap = ((mask_cur_i.float() + mask_tar_i.float()) > 1.5).float()
                mask_non_overlap = (mask_tar_i.float() - mask_overlap) > 0.5
                up_ft_cur_non_overlap = (
                    up_ft_cur[f_id][
                        mask_non_overlap.repeat(1, up_ft_cur[f_id].shape[1], 1, 1)
                    ]
                    .view(up_ft_cur[f_id].shape[1], -1)
                    .permute(1, 0)
                )
                up_ft_tar_non_overlap = (
                    up_ft_tar[f_id][
                        mask_non_overlap.repeat(1, up_ft_tar[f_id].shape[1], 1, 1)
                    ]
                    .view(up_ft_tar[f_id].shape[1], -1)
                    .permute(1, 0)
                )
                sim_non_overlap = (
                    cos(up_ft_cur_non_overlap, up_ft_tar_non_overlap) + 1.0
                ) / 2.0
                loss_edit = loss_edit + w_inpaint * sim_non_overlap.mean()
        # consistency loss
        loss_con = 0
        for f_id in range(len(up_ft_tar)):
            sim_other = (
                cos(up_ft_tar[f_id], up_ft_cur[f_id])[0][mask_other[0, 0]] + 1.0
            ) / 2.0
            loss_con = loss_con + w_content / (1 + 4 * sim_other.mean())
        loss_edit = loss_edit / len(up_ft_cur) / len(mask_cur)
        loss_con = loss_con / len(up_ft_cur)

        cond_grad_edit = torch.autograd.grad(
            loss_edit * energy_scale, latent, retain_graph=True
        )[0]
        cond_grad_con = torch.autograd.grad(loss_con * energy_scale, latent)[0]
        mask = F.interpolate(
            mask_x0[None, None],
            (cond_grad_edit[-1].shape[-2], cond_grad_edit[-1].shape[-1]),
        )
        mask = (mask > 0).to(dtype=latent.dtype)
        guidance = (
            cond_grad_edit.detach() * 4e-2 * mask
            + cond_grad_con.detach() * 4e-2 * (1 - mask)
        )
        self.feature_estimator.zero_grad()

        return guidance

    def guidance_landmark(
        self,
        mask_cur,
        mask_tar,
        latent,
        latent_noise_ref,
        t,
        up_ft_index,
        up_scale,
        context,
        context_base,
        energy_scale,
        w_edit,
        w_inpaint,
    ):
        cos = nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            up_ft_tar = self.feature_estimator(
                sample=latent_noise_ref.squeeze(2),
                timestep=t,
                up_ft_indices=up_ft_index,
                class_labels=None if self.use_cross_attn else context_base["embed"],
                encoder_hidden_states=(
                    context_base["embed"] if self.use_cross_attn else None
                ),
                encoder_attention_mask=context_base["mask"] if self.use_cross_attn else None,
            )["up_ft"]
            for f_id in range(len(up_ft_tar)):
                up_ft_tar[f_id] = F.interpolate(
                    up_ft_tar[f_id],
                    (
                        up_ft_tar[-1].shape[-2] * up_scale,
                        up_ft_tar[-1].shape[-1] * up_scale,
                    ),
                )

        latent = latent.detach().requires_grad_(True)
        up_ft_cur = self.feature_estimator(
            sample=latent,
            timestep=t,
            up_ft_indices=up_ft_index,
            class_labels=None if self.use_cross_attn else context["embed"],
            encoder_hidden_states=(
                context["embed"] if self.use_cross_attn else None
            ),
            encoder_attention_mask=context["mask"] if self.use_cross_attn else None,
        )["up_ft"]
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(
                up_ft_cur[f_id],
                (
                    up_ft_cur[-1].shape[-2] * up_scale,
                    up_ft_cur[-1].shape[-1] * up_scale,
                ),
            )

        # moving loss
        loss_edit = 0
        for f_id in range(len(up_ft_tar)):
            for mask_cur_i, mask_tar_i in zip(mask_cur, mask_tar):
                up_ft_cur_vec = (
                    up_ft_cur[f_id][
                        mask_cur_i.repeat(1, up_ft_cur[f_id].shape[1], 1, 1)
                    ]
                    .view(up_ft_cur[f_id].shape[1], -1)
                    .permute(1, 0)
                )
                up_ft_tar_vec = (
                    up_ft_tar[f_id][
                        mask_tar_i.repeat(1, up_ft_tar[f_id].shape[1], 1, 1)
                    ]
                    .view(up_ft_tar[f_id].shape[1], -1)
                    .permute(1, 0)
                )
                sim = (cos(up_ft_cur_vec, up_ft_tar_vec) + 1.0) / 2.0
                loss_edit = loss_edit + w_edit / (1 + 4 * sim.mean())
        loss_edit = loss_edit / len(up_ft_cur) / len(mask_cur)

        cond_grad_edit = torch.autograd.grad(
            loss_edit * energy_scale, latent, retain_graph=True
        )[0]
        guidance = cond_grad_edit.detach() * 4e-2
        self.feature_estimator.zero_grad()

        return guidance

    def guidance_appearance(
        self,
        mask_base_cur,
        mask_replace_cur,
        latent,
        latent_noise_ref,
        t,
        up_ft_index,
        up_scale,
        context,
        context_base,
        context_replace,
        energy_scale,
        dict_mask,
        w_edit,
        w_content,
    ):
        cos = nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            up_ft_tar_base = self.feature_estimator(
                sample=latent_noise_ref.squeeze(2)[::2],
                timestep=t,
                up_ft_indices=up_ft_index,
                class_labels=None if self.use_cross_attn else context_base["embed"],
                encoder_hidden_states=(
                    context_base["embed"] if self.use_cross_attn else None
                ),
                encoder_attention_mask=context_base["mask"] if self.use_cross_attn else None,
            )["up_ft"]
            for f_id in range(len(up_ft_tar_base)):
                up_ft_tar_base[f_id] = F.interpolate(
                    up_ft_tar_base[f_id],
                    (
                        up_ft_tar_base[-1].shape[-2] * up_scale,
                        up_ft_tar_base[-1].shape[-1] * up_scale,
                    ),
                )
        with torch.no_grad():
            up_ft_tar_replace = self.feature_estimator(
                sample=latent_noise_ref.squeeze(2)[1::2],
                timestep=t,
                up_ft_indices=up_ft_index,
                class_labels=None if self.use_cross_attn else context_replace["embed"],
                encoder_hidden_states=(
                    context_replace["embed"] if self.use_cross_attn else None
                ),
                encoder_attention_mask=context_replace["mask"] if self.use_cross_attn else None,
            )["up_ft"]
            for f_id in range(len(up_ft_tar_replace)):
                up_ft_tar_replace[f_id] = F.interpolate(
                    up_ft_tar_replace[f_id],
                    (
                        up_ft_tar_replace[-1].shape[-2] * up_scale,
                        up_ft_tar_replace[-1].shape[-1] * up_scale,
                    ),
                )
        latent = latent.detach().requires_grad_(True)
        up_ft_cur = self.feature_estimator(
            sample=latent,
            timestep=t,
            up_ft_indices=up_ft_index,
            class_labels=None if self.use_cross_attn else context["embed"],
            encoder_hidden_states=(
                context["embed"] if self.use_cross_attn else None
            ),
            encoder_attention_mask=context["mask"] if self.use_cross_attn else None,
        )["up_ft"]
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(
                up_ft_cur[f_id],
                (
                    up_ft_cur[-1].shape[-2] * up_scale,
                    up_ft_cur[-1].shape[-1] * up_scale,
                ),
            )

        # for base content
        loss_con = 0
        for f_id in range(len(up_ft_tar_base)):
            mask_cur = (1 - mask_base_cur.float()) > 0.5
            up_ft_cur_vec = (
                up_ft_cur[f_id][mask_cur.repeat(1, up_ft_cur[f_id].shape[1], 1, 1)]
                .view(up_ft_cur[f_id].shape[1], -1)
                .permute(1, 0)
            )
            up_ft_tar_vec = (
                up_ft_tar_base[f_id][
                    mask_cur.repeat(1, up_ft_tar_base[f_id].shape[1], 1, 1)
                ]
                .view(up_ft_tar_base[f_id].shape[1], -1)
                .permute(1, 0)
            )
            sim = (cos(up_ft_cur_vec, up_ft_tar_vec) + 1.0) / 2.0
            loss_con = loss_con + w_content / (1 + 4 * sim.mean())
        # for replace content
        loss_edit = 0
        for f_id in range(len(up_ft_tar_replace)):
            mask_cur = mask_base_cur
            mask_tar = mask_replace_cur
            up_ft_cur_vec = (
                up_ft_cur[f_id][mask_cur.repeat(1, up_ft_cur[f_id].shape[1], 1, 1)]
                .view(up_ft_cur[f_id].shape[1], -1)
                .permute(1, 0)
                .mean(0, keepdim=True)
            )
            up_ft_tar_vec = (
                up_ft_tar_replace[f_id][
                    mask_tar.repeat(1, up_ft_tar_replace[f_id].shape[1], 1, 1)
                ]
                .view(up_ft_tar_replace[f_id].shape[1], -1)
                .permute(1, 0)
                .mean(0, keepdim=True)
            )
            sim_all = (cos(up_ft_cur_vec, up_ft_tar_vec) + 1.0) / 2.0
            loss_edit = loss_edit + w_edit / (1 + 4 * sim_all.mean())

        cond_grad_con = torch.autograd.grad(
            loss_con * energy_scale, latent, retain_graph=True
        )[0]
        cond_grad_edit = torch.autograd.grad(loss_edit * energy_scale, latent)[0]
        mask = F.interpolate(
            mask_base_cur.float(),
            (cond_grad_edit[-1].shape[-2], cond_grad_edit[-1].shape[-1]),
        )
        mask = (mask > 0).to(dtype=latent.dtype)
        guidance = (
            cond_grad_con.detach() * (1 - mask) * 4e-2
            + cond_grad_edit.detach() * mask * 4e-2
        )
        self.feature_estimator.zero_grad()

        return guidance

    def guidance_mix(
        self,
        mask_base_cur,
        mask_replace_cur,
        latent,
        latent_noise_ref,
        t,
        up_ft_index,
        up_scale,
        context,
        context_base,
        context_replace,
        energy_scale,
        dict_mask,
        w_edit,
        w_content,
    ):
        cos = nn.CosineSimilarity(dim=1)
        # pcpt_loss_fn = LPIPS()
        with torch.no_grad():
            up_ft_tar_base = self.feature_estimator(
                sample=latent_noise_ref.squeeze(2)[::2],
                timestep=t,
                up_ft_indices=up_ft_index,
                class_labels=None if self.use_cross_attn else context_base["embed"],
                encoder_hidden_states=context_base["embed"] if self.use_cross_attn else None,
                encoder_attention_mask=context_base["mask"] if self.use_cross_attn else None,
            )["up_ft"]
            for f_id in range(len(up_ft_tar_base)):
                up_ft_tar_base[f_id] = F.interpolate(
                    up_ft_tar_base[f_id],
                    (
                        up_ft_tar_base[-1].shape[-2] * up_scale,
                        up_ft_tar_base[-1].shape[-1] * up_scale,
                    ),
                )
        with torch.no_grad():
            up_ft_tar_replace = self.feature_estimator(
                sample=latent_noise_ref.squeeze(2)[1::2],
                timestep=t,
                up_ft_indices=up_ft_index,
                class_labels=None if self.use_cross_attn else context_replace["embed"],
                encoder_hidden_states=context_replace["embed"] if self.use_cross_attn else None,
                encoder_attention_mask=context_replace["mask"] if self.use_cross_attn else None,
            )["up_ft"]
            for f_id in range(len(up_ft_tar_replace)):
                up_ft_tar_replace[f_id] = F.interpolate(
                    up_ft_tar_replace[f_id],
                    (
                        up_ft_tar_replace[-1].shape[-2] * up_scale,
                        up_ft_tar_replace[-1].shape[-1] * up_scale,
                    ),
                )

        latent = latent.detach().requires_grad_(True)

        up_ft_cur = self.feature_estimator(
            sample=latent,
            timestep=t,
            up_ft_indices=up_ft_index,
            class_labels=None if self.use_cross_attn else context["embed"],
            encoder_hidden_states=context["embed"] if self.use_cross_attn else None,
            encoder_attention_mask=context["mask"] if self.use_cross_attn else None,
        )["up_ft"]
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(
                up_ft_cur[f_id],
                (
                    up_ft_cur[-1].shape[-2] * up_scale,
                    up_ft_cur[-1].shape[-1] * up_scale,
                ),
            )
        # for base content
        # loss_con = 0
        # for f_id in range(len(up_ft_tar_base)):
        #     mask_cur = (1-mask_base_cur.float())>0.5
        #     up_ft_cur_vec = up_ft_cur[f_id][mask_cur.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
        #     up_ft_tar_vec = up_ft_tar_base[f_id][mask_cur.repeat(1,up_ft_tar_base[f_id].shape[1],1,1)].view(up_ft_tar_base[f_id].shape[1], -1).permute(1,0)
        #     sim = (cos(up_ft_cur_vec, up_ft_tar_vec)+1.)/2.
        #     loss_con = loss_con + w_content/(1+4*sim.mean())
        loss_con = 0
        for f_id in range(len(up_ft_tar_base)):
            mask_cur = (1 - mask_base_cur.float()) > 0.5
            mask_cur_p = mask_base_cur.float() > 0.5
            up_ft_cur_vec = (
                up_ft_cur[f_id][mask_cur.repeat(1, up_ft_cur[f_id].shape[1], 1, 1)]
                .view(up_ft_cur[f_id].shape[1], -1)
                .permute(1, 0)
            )
            up_ft_tar_vec = (
                up_ft_tar_base[f_id][
                    mask_cur.repeat(1, up_ft_tar_base[f_id].shape[1], 1, 1)
                ]
                .view(up_ft_tar_base[f_id].shape[1], -1)
                .permute(1, 0)
            )
            sim = (cos(up_ft_cur_vec, up_ft_tar_vec) + 1.0) / 2.0
            loss_con = loss_con + w_content / (1 + 4 * sim.mean())
        # for replace content
        loss_edit = 0
        for f_id in range(len(up_ft_tar_replace)):
            mask_cur = mask_base_cur

            up_ft_cur_vec = (
                up_ft_cur[f_id][mask_cur.repeat(1, up_ft_cur[f_id].shape[1], 1, 1)]
                .view(up_ft_cur[f_id].shape[1], -1)
                .permute(1, 0)
            )
            up_ft_tar_vec = (
                up_ft_tar_replace[f_id][
                    mask_replace_cur.repeat(1, up_ft_tar_replace[f_id].shape[1], 1, 1)
                ]
                .view(up_ft_tar_replace[f_id].shape[1], -1)
                .permute(1, 0)
            )
            up_ft_tar_vec_b = (
                up_ft_tar_base[f_id][
                    mask_cur.repeat(1, up_ft_tar_base[f_id].shape[1], 1, 1)
                ]
                .view(up_ft_tar_base[f_id].shape[1], -1)
                .permute(1, 0)
            )
            # sim_all=0.8*((cos(up_ft_cur_vec, up_ft_tar_vec_b)+1.)/2.) + 0.2*((cos(up_ft_cur_vec, up_ft_tar_vec)+1.)/2.)
            # # sim_all=0.7*((cos(up_ft_cur_vec, up_ft_tar_vec)+1.)/2.)+0.3*((cos(up_ft_cur_vec, up_ft_tar_vec_b)+1.)/2.)
            # loss_edit=loss_edit+w_edit/(1+4*sim_all.mean())
            # NOTE: try to use Harmonic mean
            sim_base = (cos(up_ft_cur_vec, up_ft_tar_vec_b) + 1.0) / 2.0
            sim_tar = (cos(up_ft_cur_vec, up_ft_tar_vec) + 1.0) / 2.0
            loss_edit = (
                loss_edit
                + w_content / (1 + 4 * sim_base.mean())
                + w_edit / (1 + 4 * sim_tar.mean())
            )  # NOTE empirically 0.7 is a good bg to all ratio

        cond_grad_con = torch.autograd.grad(
            loss_con * energy_scale, latent, retain_graph=True
        )[0]
        cond_grad_edit = torch.autograd.grad(loss_edit * energy_scale, latent)[0]
        mask = F.interpolate(
            mask_base_cur.float(),
            (cond_grad_edit[-1].shape[-2], cond_grad_edit[-1].shape[-1]),
        )
        mask = (mask > 0).to(dtype=latent.dtype)
        guidance = (
            cond_grad_con.detach() * (1 - mask) * 4e-2
            + cond_grad_edit.detach() * mask * 4e-2
        )
        self.feature_estimator.zero_grad()

        return guidance

    def guidance_paste(
        self,
        mask_base_cur,
        mask_replace_cur,
        latent,
        latent_noise_ref,
        t,
        up_ft_index,
        up_scale,
        context,
        context_base,
        context_replace,
        energy_scale,
        dict_mask,
        w_edit,
        w_content,
    ):
        cos = nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            up_ft_tar_base = self.feature_estimator(
                sample=latent_noise_ref.squeeze(2)[::2],
                timestep=t,
                up_ft_indices=up_ft_index,
                class_labels=None if self.use_cross_attn else context_base["embed"],
                encoder_hidden_states=context_base["embed"] if self.use_cross_attn else None,
                encoder_attention_mask=context_base["mask"] if self.use_cross_attn else None,
            )["up_ft"]
            for f_id in range(len(up_ft_tar_base)):
                up_ft_tar_base[f_id] = F.interpolate(
                    up_ft_tar_base[f_id],
                    (
                        up_ft_tar_base[-1].shape[-2] * up_scale,
                        up_ft_tar_base[-1].shape[-1] * up_scale,
                    ),
                )
        with torch.no_grad():
            up_ft_tar_replace = self.feature_estimator(
                sample=latent_noise_ref.squeeze(2)[1::2],
                timestep=t,
                up_ft_indices=up_ft_index,
                class_labels=None if self.use_cross_attn else context_replace["embed"],
                encoder_hidden_states=context_replace["embed"] if self.use_cross_attn else None,
                encoder_attention_mask=context_replace["mask"] if self.use_cross_attn else None,
            )["up_ft"]
            for f_id in range(len(up_ft_tar_replace)):
                up_ft_tar_replace[f_id] = F.interpolate(
                    up_ft_tar_replace[f_id],
                    (
                        up_ft_tar_replace[-1].shape[-2] * up_scale,
                        up_ft_tar_replace[-1].shape[-1] * up_scale,
                    ),
                )

        latent = latent.detach().requires_grad_(True)

        up_ft_cur = self.feature_estimator(
            sample=latent,
            timestep=t,
            up_ft_indices=up_ft_index,
            class_labels=None if self.use_cross_attn else context["embed"],
            encoder_hidden_states=context["embed"] if self.use_cross_attn else None,
            encoder_attention_mask=context["mask"] if self.use_cross_attn else None,
        )["up_ft"]
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(
                up_ft_cur[f_id],
                (
                    up_ft_cur[-1].shape[-2] * up_scale,
                    up_ft_cur[-1].shape[-1] * up_scale,
                ),
            )
        # for base content
        loss_con = 0
        for f_id in range(len(up_ft_tar_base)):
            mask_cur = (1 - mask_base_cur.float()) > 0.5
            mask_cur_p = mask_base_cur.float() > 0.5
            up_ft_cur_vec = (
                up_ft_cur[f_id][mask_cur.repeat(1, up_ft_cur[f_id].shape[1], 1, 1)]
                .view(up_ft_cur[f_id].shape[1], -1)
                .permute(1, 0)
            )
            up_ft_tar_vec = (
                up_ft_tar_base[f_id][
                    mask_cur.repeat(1, up_ft_tar_base[f_id].shape[1], 1, 1)
                ]
                .view(up_ft_tar_base[f_id].shape[1], -1)
                .permute(1, 0)
            )
            sim = (cos(up_ft_cur_vec, up_ft_tar_vec) + 1.0) / 2.0
            loss_con = loss_con + w_content / (1 + 4 * sim.mean())
        # for replace content
        loss_edit = 0
        for f_id in range(len(up_ft_tar_replace)):
            mask_cur = mask_base_cur

            up_ft_cur_vec = (
                up_ft_cur[f_id][mask_cur.repeat(1, up_ft_cur[f_id].shape[1], 1, 1)]
                .view(up_ft_cur[f_id].shape[1], -1)
                .permute(1, 0)
            )
            up_ft_tar_vec = (
                up_ft_tar_replace[f_id][
                    mask_replace_cur.repeat(1, up_ft_tar_replace[f_id].shape[1], 1, 1)
                ]
                .view(up_ft_tar_replace[f_id].shape[1], -1)
                .permute(1, 0)
            )
            sim_all = (cos(up_ft_cur_vec, up_ft_tar_vec) + 1.0) / 2.0
            loss_edit = loss_edit + w_edit / (1 + 4 * sim_all.mean())

        cond_grad_con = torch.autograd.grad(
            loss_con * energy_scale, latent, retain_graph=True
        )[0]

        cond_grad_edit = torch.autograd.grad(loss_edit * energy_scale, latent)[0]
        mask = F.interpolate(
            mask_base_cur.float(),
            (cond_grad_edit[-1].shape[-2], cond_grad_edit[-1].shape[-1]),
        )
        mask = (mask > 0).to(dtype=latent.dtype)
        guidance = (
            cond_grad_con.detach() * (1 - mask) * 4e-2
            + cond_grad_edit.detach() * mask * 4e-2
        )
        self.feature_estimator.zero_grad()

        return guidance

    def guidance_remove(
        self,
        mask_base_cur,
        mask_replace_cur,
        latent,
        latent_noise_ref,
        t,
        up_ft_index,
        up_scale,
        context,
        context_base,
        context_replace,
        energy_scale,
        dict_mask,
        w_edit,
        w_contrast,
        w_content,
    ):
        cos = nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            up_ft_tar_base = self.feature_estimator(
                sample=latent_noise_ref.squeeze(2)[::2],
                timestep=t,
                up_ft_indices=up_ft_index,
                class_labels=None if self.use_cross_attn else context_base["embed"],
                encoder_hidden_states=context_base["embed"] if self.use_cross_attn else None,
                encoder_attention_mask=context_base["mask"] if self.use_cross_attn else None,
            )["up_ft"]
            for f_id in range(len(up_ft_tar_base)):
                up_ft_tar_base[f_id] = F.interpolate(
                    up_ft_tar_base[f_id],
                    (
                        up_ft_tar_base[-1].shape[-2] * up_scale,
                        up_ft_tar_base[-1].shape[-1] * up_scale,
                    ),
                )
                # up_ft_tar_base[f_id] = normalize_along_channel(up_ft_tar_base[f_id])  # No improvement observed

        with torch.no_grad():
            up_ft_tar_replace = self.feature_estimator(
                sample=latent_noise_ref.squeeze(2)[1::2],
                timestep=t,
                up_ft_indices=up_ft_index,
                class_labels=None if self.use_cross_attn else context_replace["embed"],
                encoder_hidden_states=context_replace["embed"] if self.use_cross_attn else None,
                encoder_attention_mask=context_replace["mask"] if self.use_cross_attn else None,
            )["up_ft"]
            for f_id in range(len(up_ft_tar_replace)):
                up_ft_tar_replace[f_id] = F.interpolate(
                    up_ft_tar_replace[f_id],
                    (
                        up_ft_tar_replace[-1].shape[-2] * up_scale,
                        up_ft_tar_replace[-1].shape[-1] * up_scale,
                    ),
                )
                # up_ft_tar_replace[f_id] = normalize_along_channel(up_ft_tar_replace[f_id])

        latent = latent.detach().requires_grad_(True)
        up_ft_cur = self.feature_estimator(
            sample=latent,
            timestep=t,
            up_ft_indices=up_ft_index,
            class_labels=None if self.use_cross_attn else context["embed"],
            encoder_hidden_states=context["embed"] if self.use_cross_attn else None,
            encoder_attention_mask=context["mask"] if self.use_cross_attn else None,
        )["up_ft"]
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(
                up_ft_cur[f_id],
                (
                    up_ft_cur[-1].shape[-2] * up_scale,
                    up_ft_cur[-1].shape[-1] * up_scale,
                ),
            )
            # up_ft_cur[f_id] = normalize_along_channel(up_ft_cur[f_id])

        # for base content
        loss_con = 0
        for f_id in range(len(up_ft_tar_base)):
            mask_cur = (1 - mask_base_cur.float()) > 0.5
            up_ft_cur_vec = (
                up_ft_cur[f_id][mask_cur.repeat(1, up_ft_cur[f_id].shape[1], 1, 1)]
                .view(up_ft_cur[f_id].shape[1], -1)
                .permute(1, 0)
            )
            up_ft_tar_vec = (
                up_ft_tar_base[f_id][
                    mask_cur.repeat(1, up_ft_tar_base[f_id].shape[1], 1, 1)
                ]
                .view(up_ft_tar_base[f_id].shape[1], -1)
                .permute(1, 0)
            )
            sim = (cos(up_ft_cur_vec, up_ft_tar_vec) + 1.0) / 2.0
            loss_con = loss_con + w_content / (1 + 4 * sim.mean())
        # for replace content
        loss_edit = 0
        for f_id in range(len(up_ft_tar_replace)):
            mask_cur = mask_base_cur

            # NOTE: Uncomment to get Global time&freq
            # up_ft_cur_vec = up_ft_cur[f_id][mask_cur.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            # up_ft_tar_vec = up_ft_tar_replace[f_id][mask_replace_cur.repeat(1,up_ft_tar_replace[f_id].shape[1],1,1)].view(up_ft_tar_replace[f_id].shape[1], -1).permute(1,0)
            # sim_all=((cos(up_ft_cur_vec.mean(0,keepdim=True), up_ft_tar_vec.mean(0,keepdim=True))+1.)/2.)

            # Get a vec along time axis (global time)
            up_ft_cur_vec_masked_sum = torch.sum(
                up_ft_cur[f_id] * mask_cur.repeat(1, up_ft_cur[f_id].shape[1], 1, 1),
                dim=-2,
            )
            up_ft_tar_vec_masked_sum = torch.sum(
                up_ft_tar_replace[f_id]
                * mask_replace_cur.repeat(1, up_ft_tar_replace[f_id].shape[1], 1, 1),
                dim=-2,
            )  # feature for reference audio

            mask_sum = torch.sum(mask_cur, dim=-2)

            up_ft_cur_vec = (
                (up_ft_cur_vec_masked_sum / (mask_sum + 1e-8))
                .view(up_ft_cur[f_id].shape[1], -1)
                .permute(1, 0)
            )
            up_ft_tar_vec = (
                (up_ft_tar_vec_masked_sum / (mask_sum + 1e-8))
                .view(up_ft_tar_replace[f_id].shape[1], -1)
                .permute(1, 0)
            )
            sim_edit_contrast = (cos(up_ft_cur_vec, up_ft_tar_vec) + 1.0) / 2.0

            # NOTE: begin to modify energy func
            up_ft_cur_vec_base = (
                up_ft_cur[f_id][mask_cur.repeat(1, up_ft_cur[f_id].shape[1], 1, 1)]
                .view(up_ft_cur[f_id].shape[1], -1)
                .permute(1, 0)
            )
            mask_cur_con = mask_cur
            up_ft_tar_vec_base = (
                up_ft_tar_base[f_id][
                    mask_cur_con.repeat(1, up_ft_tar_base[f_id].shape[1], 1, 1)
                ]
                .view(up_ft_tar_base[f_id].shape[1], -1)
                .permute(1, 0)
            )
            # # local
            # sim_edit_consist = (cos(up_ft_cur_vec_base, up_ft_tar_vec_base) + 1.0) / 2.0
            # global
            sim_edit_consist=((cos(up_ft_cur_vec_base.mean(0, keepdim=True), up_ft_tar_vec_base.mean(0, keepdim=True))+1.)/2.)

            # loss_edit = loss_edit - w_edit/(1+4*sim_all.mean()) + w_edit/(1+4*sim_edit_consist.mean()) # NOTE: decrease sim
            # loss_edit = loss_edit + w_edit*sim_all.mean()
            # loss_edit = loss_edit - 0.1*w_edit/(1+4*sim_all.mean()) + w_edit/(1+4*sim_edit_consist.mean())
            # loss_edit = loss_edit - 0.5*w_edit/(1+4*sim_all.mean())  # NOTE: this only affect local features not semantic
            # loss_edit = loss_edit + 0.005*w_edit*sim_all.mean() + w_edit/(1+4*sim_edit_consist.mean()) # NOTE: local
            loss_edit = (
                loss_edit
                + w_contrast * sim_edit_contrast.mean()
                + w_edit / (1 + 4 * sim_edit_consist.mean())
            )  # NOTE: local

        cond_grad_con = torch.autograd.grad(
            loss_con * energy_scale, latent, retain_graph=True
        )[0]
        cond_grad_edit = torch.autograd.grad(loss_edit * energy_scale, latent)[0]
        mask = F.interpolate(
            mask_base_cur.float(),
            (cond_grad_edit[-1].shape[-2], cond_grad_edit[-1].shape[-1]),
        )
        mask = (mask > 0).to(dtype=latent.dtype)
        guidance = (
            cond_grad_con.detach() * (1 - mask) * 4e-2
            + cond_grad_edit.detach() * mask * 4e-2
        )
        self.feature_estimator.zero_grad()

        return guidance


    def guidance_style_transfer(
        self,
        mask_base_cur,
        mask_replace_cur,
        latent,
        latent_noise_ref,
        t,
        up_ft_index,
        up_scale,
        context,
        context_base,
        energy_scale,
        dict_mask,
        w_edit,
        w_content,
    ):
        cos = nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            up_ft_tar_base = self.feature_estimator(
                sample=latent_noise_ref.squeeze(2),
                timestep=t,
                up_ft_indices=up_ft_index,
                class_labels=None if self.use_cross_attn else context_base["embed"],
                encoder_hidden_states=context_base["embed"] if self.use_cross_attn else None,
                encoder_attention_mask=context_base["mask"] if self.use_cross_attn else None,
            )["up_ft"]
            for f_id in range(len(up_ft_tar_base)):
                up_ft_tar_base[f_id] = F.interpolate(
                    up_ft_tar_base[f_id],
                    (
                        up_ft_tar_base[-1].shape[-2] * up_scale,
                        up_ft_tar_base[-1].shape[-1] * up_scale,
                    ),
                )

        latent = latent.detach().requires_grad_(True)
        up_ft_cur = self.feature_estimator(
            sample=latent,
            timestep=t,
            up_ft_indices=up_ft_index,
            class_labels=None if self.use_cross_attn else context["embed"],
            encoder_hidden_states=context["embed"] if self.use_cross_attn else None,
            encoder_attention_mask=context["mask"] if self.use_cross_attn else None,
        )["up_ft"]
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(
                up_ft_cur[f_id],
                (up_ft_cur[-1].shape[-2] * up_scale, up_ft_cur[-1].shape[-1] * up_scale),
            )
        # for base content
        loss_con = 0
        for f_id in range(len(up_ft_tar_base)):
            mask_cur = (1 - mask_base_cur.float()) > 0.5
            mask_cur_p = mask_base_cur.float() > 0.5
            up_ft_cur_vec = (
                up_ft_cur[f_id][mask_cur.repeat(1, up_ft_cur[f_id].shape[1], 1, 1)]
                .view(up_ft_cur[f_id].shape[1], -1)
                .permute(1, 0)
            )
            up_ft_tar_vec = (
                up_ft_tar_base[f_id][
                    mask_cur.repeat(1, up_ft_tar_base[f_id].shape[1], 1, 1)
                ]
                .view(up_ft_tar_base[f_id].shape[1], -1)
                .permute(1, 0)
            )
            sim = (cos(up_ft_cur_vec, up_ft_tar_vec) + 1.0) / 2.0
            loss_con = loss_con + w_content / (1 + 4 * sim.mean())

        cond_grad_con = torch.autograd.grad(
            loss_con * energy_scale, latent, retain_graph=True
        )[0]
        
        mask = F.interpolate(
            mask_base_cur.float(),
            (cond_grad_con[-1].shape[-2], cond_grad_con[-1].shape[-1]),
        )
        mask = (mask > 0).to(dtype=latent.dtype)
        guidance = cond_grad_con.detach() * (1 - mask) * 4e-2
        self.feature_estimator.zero_grad()

        return guidance

    def _encode_text(self, text_input: str):
        text_input = self.tokenizer(
            [text_input],
            padding="max_length",
            max_length=self.tokenizer.model_max_length, # NOTE 77
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attn_mask = text_input.input_ids.to(self._device), text_input.attention_mask.to(self._device)

        text_embeddings = self.text_encoder(input_ids,attention_mask=attn_mask)[0]
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        boolean_attn_mask = (attn_mask == 1).to(self._device)

        return {"embed": text_embeddings, "mask": boolean_attn_mask}
    
    def _stack_text(self, text_input0, text_input1):
        text_embs0, text_mask0 = text_input0["embed"], text_input0["mask"]
        text_embs1, text_mask1 = text_input1["embed"], text_input1["mask"]
        out_embs = torch.cat(
            [text_embs0.expand(*text_embs1.shape), text_embs1]
        )
        out_mask = torch.cat(
            [text_mask0.expand(*text_mask1.shape), text_mask1]
            )
        
        return {"embed": out_embs, "mask": out_mask}