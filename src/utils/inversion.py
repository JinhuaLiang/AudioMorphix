import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Union, Tuple, List, Callable, Dict


class DDIMInversion:
    def __init__(self, model, NUM_DDIM_STEPS):
        self.model = model
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.NUM_DDIM_STEPS = NUM_DDIM_STEPS
        self.prompt = None

    def next_step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
        prediction_type: str = "v_prediction",
    ):
        timestep, next_timestep = (
            min(
                timestep
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps,
                999,
            ),
            timestep,
        )
        alpha_prod_t = (
            self.scheduler.alphas_cumprod[timestep]
            if timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t

        if prediction_type == "epsilon":
            next_original_sample = (
                sample - beta_prod_t**0.5 * model_output
            ) / alpha_prod_t**0.5
            next_epsilon = model_output
        elif prediction_type == "v_prediction":
            next_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            next_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {prediction_type} must be one of `epsilon` or"
                " `v_prediction`"
            )
        
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * next_epsilon
        next_sample = (
            alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
        )
        return next_sample

    def get_noise_pred_single(
        self, latents, t, cond_embeddings, cond_masks, iter_cur, save_kv=True, mode="drag"
    ):
        boolean_cond_masks = (cond_masks == 1).to(cond_masks.device)

        try:
            noise_pred = self.model.unet(
                latents,
                t,
                encoder_hidden_states=(
                    cond_embeddings if self.model.use_cross_attn else None
                ),
                class_labels=None if self.model.use_cross_attn else cond_embeddings,
                encoder_attention_mask=boolean_cond_masks if self.model.use_cross_attn else None,
                iter_cur=iter_cur,
                mode=mode,
                save_kv=save_kv,
            )["sample"]

        except TypeError as e:
            print(f"Warning: {e}")
            noise_pred = self.model.unet(
                latents,
                t,
                encoder_hidden_states=(
                    cond_embeddings if self.model.use_cross_attn else None
                ),
                class_labels=None if self.model.use_cross_attn else cond_embeddings,
                encoder_attention_mask=boolean_cond_masks if self.model.use_cross_attn else None,
            )["sample"]

        return noise_pred

    @torch.no_grad()
    def init_prompt(self, prompt: str, emb_im=None):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = self.model.text_encoder.device
        
        if not isinstance(prompt, list):
            prompt = [prompt]

        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attn_masks = text_input.input_ids.to(device), text_input.attention_mask.to(device)
     
        text_embeddings = self.model.text_encoder(
            input_ids, attention_mask=attn_masks,
        )[0]
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        if emb_im is not None:
            raise NotImplementedError
        
            self.text_embeddings = torch.cat([text_embeddings, emb_im], dim=1)
        else:
            self.text_embeddings = text_embeddings
            self.text_masks = attn_masks

        self.prompt = prompt
        

    @torch.no_grad()
    def ddim_loop(self, latent, save_kv=True, mode="drag", prediction_type="v_prediction"):
        cond_embeddings = self.text_embeddings
        cond_masks = self.text_masks
        all_latent = [latent]
        latent = latent.clone().detach()
        print("DDIM Inversion:")
        for i in tqdm(range(self.NUM_DDIM_STEPS)):
            t = self.model.scheduler.timesteps[
                len(self.model.scheduler.timesteps) - i - 1
            ]
            noise_pred = self.get_noise_pred_single(
                latent,
                t,
                cond_embeddings,
                cond_masks,
                iter_cur=len(self.model.scheduler.timesteps) - i - 1,
                save_kv=save_kv,
                mode=mode,
            )
            latent = self.next_step(noise_pred, t, latent, prediction_type=prediction_type)
            all_latent.append(latent)

        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    def invert(self, ddim_latents, prompt: str, emb_im=None, save_kv=True, mode="drag", prediction_type="v_prediction"):
        self.init_prompt(prompt, emb_im=emb_im)
        ddim_latents = self.ddim_loop(ddim_latents, save_kv=save_kv, mode=mode, prediction_type=prediction_type)
        return ddim_latents
