import gc
import os
import yaml
import inspect
import torch
import torch.nn as nn
import numpy as np
from diffusers import DDIMScheduler
from PIL import Image

# from basicsr.utils import tensor2img
from diffusers import AutoencoderKL
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    AutoTokenizer,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
    ClapTextModelWithProjection,
    RobertaTokenizer,
    RobertaTokenizerFast,
    SpeechT5HifiGan,
)
from diffusers.utils.import_utils import is_xformers_available


from src.module.unet.unet_2d_condition import (
    CustomUNet2DConditionModel,
    UNet2DConditionModel,
)
from src.module.unet.estimator import _UNet2DConditionModel
from src.utils.inversion import DDIMInversion
from src.module.unet.attention_processor import (
    IPAttnProcessor,
    AttnProcessor,
    Resampler,
)
from src.model.sampler import Sampler
from src.utils.audio_processing import extract_fbank, wav_to_fbank, TacotronSTFT, maybe_add_dimension
import sys
sys.path.append("src/module/tango")
from tools.torch_tools import wav_to_fbank as tng_wav_to_fbank


CWD = os.getcwd()


class TangoPipeline:
    def __init__(
        self,
        sd_id="declare-lab/tango",
        NUM_DDIM_STEPS=100,
        precision=torch.float32,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs,
    ):
        import sys
        import json
        import torch
        from huggingface_hub import snapshot_download

        sys.path.append("./src/module/tango")
        from tango2.models import AudioDiffusion
        from audioldm.audio.stft import TacotronSTFT as tng_TacotronSTFT
        from audioldm.variational_autoencoder import AutoencoderKL

        path = snapshot_download(repo_id=sd_id)
        vae_config = json.load(open("{}/vae_config.json".format(path)))
        stft_config = json.load(open("{}/stft_config.json".format(path)))
        main_config = json.load(open("{}/main_config.json".format(path)))
        main_config["unet_model_config_path"] = os.path.join(
            CWD, "src/module/tango", main_config["unet_model_config_path"]
        )

        unet = self._set_unet2dconditional_model(
            CustomUNet2DConditionModel,
            unet_model_name=main_config["unet_model_name"],
            unet_model_config_path=main_config["unet_model_config_path"],
        ).to(device)
        feature_estimator = self._set_unet2dconditional_model(
            _UNet2DConditionModel,
            unet_model_name=main_config["unet_model_name"],
            unet_model_config_path=main_config["unet_model_config_path"],
        ).to(device)

        ##### Load pretrained model #####
        vae = AutoencoderKL(**vae_config).to(device)
        vae.dtype = torch.float32  # avoid attribute missing
        stft = tng_TacotronSTFT(**stft_config).to(device)
        model = AudioDiffusion(**main_config).to(device)
        model.unet = unet  # replace unet with the custom unet

        vae_weights = torch.load(
            "{}/pytorch_model_vae.bin".format(path), map_location=device
        )
        stft_weights = torch.load(
            "{}/pytorch_model_stft.bin".format(path), map_location=device
        )
        main_weights = torch.load(
            "{}/pytorch_model_main.bin".format(path), map_location=device
        )

        vae.load_state_dict(vae_weights)
        stft.load_state_dict(stft_weights)
        model.load_state_dict(main_weights)

        unet_weights = {".".join(layer.split(".")[1:]): param for layer, param in model.named_parameters() if "unet" in layer}
        feature_estimator.load_state_dict(unet_weights)

        vae.eval()
        stft.eval()
        model.eval()
        feature_estimator.eval()

        # Free memeory
        del vae_weights
        del stft_weights
        del main_weights
        del unet_weights

        feature_estimator.scheduler = DDIMScheduler.from_pretrained(
            main_config["scheduler_name"], subfolder="scheduler"
        )

        # Create pipeline for audio editing
        onestep_pipe = Sampler(
            vae=vae,
            tokenizer=model.tokenizer,
            text_encoder=model.text_encoder,
            unet=model.unet,
            feature_estimator=feature_estimator,
            scheduler=DDIMScheduler.from_pretrained(
                main_config["scheduler_name"], subfolder="scheduler"
            ),
            device=device,
            precision=precision,
        )
        onestep_pipe.use_cross_attn = True

        gc.collect()
        onestep_pipe.enable_attention_slicing()
        if is_xformers_available():
            onestep_pipe.feature_estimator.enable_xformers_memory_efficient_attention()
            onestep_pipe.enable_xformers_memory_efficient_attention()
        self.pipe = onestep_pipe
        self.fn_STFT = stft

        self.vae_scale_factor = vae_config["ddconfig"]["ch_mult"][-1]
        self.NUM_DDIM_STEPS = NUM_DDIM_STEPS
        self.num_tokens = 512  # flant5
        self.precision = precision
        self.device = device

        # self.load_adapter()  # replace the 1-st self-attn layer with cross-attn difference trajactory

    def _set_unet2dconditional_model(
        self,
        cls_obj: UNet2DConditionModel,
        *,
        unet_model_name=None,
        unet_model_config_path=None,
    ):
        assert (
            unet_model_name is not None or unet_model_config_path is not None
        ), "Either UNet pretrain model name or a config file path is required"

        if unet_model_config_path:
            unet_config = cls_obj.load_config(unet_model_config_path)
            unet = cls_obj.from_config(unet_config, subfolder="unet")
            unet.set_from = "random"
        else:
            unet = cls_obj.from_pretrained(unet_model_name, subfolder="unet")
            unet.set_from = "pre-trained"
            unet.group_in = nn.Sequential(nn.Linear(8, 512), nn.Linear(512, 4))
            unet.group_out = nn.Sequential(nn.Linear(4, 512), nn.Linear(512, 8))

        return unet

    @torch.no_grad()
    def decode_latents(self, latents):
        return self.pipe.vae.decode_first_stage(latents)

    @torch.no_grad()
    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        return self.pipe.vae.decode_to_waveform(mel_spectrogram)

    def get_fbank(self, audio_or_path, stft_cfg, return_intermediate=False):
        r"""Helper function to get fbank from audio file."""
        if isinstance(audio_or_path, torch.Tensor):
            return maybe_add_dimension(audio_or_path, 4)
        
        if isinstance(audio_or_path, str):
            fbank, log_stft, wav = tng_wav_to_fbank(
                [audio_or_path],
                fn_STFT=self.fn_STFT,
                target_length=stft_cfg.filter_length,
            )
            fbank = maybe_add_dimension(fbank, 4)  # (B,C,T,F)

            if return_intermediate:
                return fbank, log_stft, wav
            
            return fbank

    @torch.no_grad()
    def encode_fbank(self, fbank):
        return self.pipe.vae.get_first_stage_encoding(
            self.pipe.vae.encode_first_stage(fbank)
        )

    @torch.no_grad()
    def fbank2latent(self, fbank):
        latent = self.encode_fbank(fbank)
        return latent

    def ddim_inv(self, latent, prompt, emb_im=None, save_kv=True, mode="mix", prediction_type="v_prediction"):
        ddim_inv = DDIMInversion(model=self.pipe, NUM_DDIM_STEPS=self.NUM_DDIM_STEPS)
        ddim_latents = ddim_inv.invert(
            ddim_latents=latent.unsqueeze(2), prompt=prompt, emb_im=emb_im,
            save_kv=save_kv, mode=mode, prediction_type=prediction_type,
        )
        return ddim_latents

    def init_proj(self, precision):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to("cuda", dtype=precision)
        return image_proj_model

    def load_adapter(self):
        scale = 1.0
        attn_procs = {}
        for name in self.pipe.unet.attn_processors.keys():
            cross_attention_dim = None
            if name.startswith("mid_block"):
                hidden_size = self.pipe.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.pipe.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.pipe.unet.config.block_out_channels[block_id]
            # Only the first self-attention should be used for cross-attend different trojactory
            if name.endswith("attn1.processor"):
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=scale,
                    num_tokens=self.num_tokens,
                ).to("cuda", dtype=self.precision)
        self.pipe.unet.set_attn_processor(attn_procs)


class AudioLDMPipeline:
    def __init__(
        self,
        sd_id="cvssp/audioldm-l-full",
        ip_id="cvssp/audioldm-l-full",
        NUM_DDIM_STEPS=50,
        precision=torch.float32,
        ip_scale=0,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        onestep_pipe = Sampler(
            vae=AutoencoderKL.from_pretrained(
                sd_id, subfolder="vae", torch_dtype=precision
            ),
            tokenizer=RobertaTokenizerFast.from_pretrained(
                sd_id, subfolder="tokenizer"
            ),
            text_encoder=ClapTextModelWithProjection.from_pretrained(
                sd_id, subfolder="text_encoder", torch_dtype=precision
            ),
            unet=CustomUNet2DConditionModel.from_pretrained(
                sd_id, subfolder="unet", torch_dtype=precision
            ),
            feature_estimator=_UNet2DConditionModel.from_pretrained(
                sd_id,
                subfolder="unet",
                vae=None,
                text_encoder=None,
                tokenizer=None,
                scheduler=DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler"),
                safety_checker=None,
                feature_extractor=None,
            ),
            scheduler=DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler"),
            device=device,
            precision=precision,
        )

        onestep_pipe.vocoder = SpeechT5HifiGan.from_pretrained(
            sd_id, subfolder="vocoder", torch_dtype=precision
        )

        onestep_pipe.use_cross_attn = False
        gc.collect()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        onestep_pipe = onestep_pipe.to(device)
        onestep_pipe.vocoder.to(device)
        onestep_pipe.enable_attention_slicing()
        if is_xformers_available():
            onestep_pipe.feature_estimator.enable_xformers_memory_efficient_attention()
            onestep_pipe.enable_xformers_memory_efficient_attention()

        self.pipe = onestep_pipe
        self.vae_scale_factor = 2 ** (len(self.pipe.vae.config.block_out_channels) - 1)
        self.NUM_DDIM_STEPS = NUM_DDIM_STEPS
        self.precision = precision
        self.device = device
        self.num_tokens = 64

        # This is fixed as per pretrained model
        self.fn_STFT = TacotronSTFT(
            filter_length=1024,
            hop_length=160,
            win_length=1024,
            n_mel_channels=64,
            sampling_rate=16000,
            mel_fmin=0,
            mel_fmax=8000,
        )

        # self.load_adapter()

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / self.pipe.vae.config.scaling_factor * latents
        mel_spectrogram = self.pipe.vae.decode(latents).sample
        return mel_spectrogram

    @torch.no_grad()
    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.pipe.vocoder(
            mel_spectrogram.to(device=self.device, dtype=self.precision)
        )
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform.cpu().float()
        return waveform

    @torch.no_grad()
    def fbank2latent(self, fbank):
        latent = self.encode_fbank(fbank)
        return latent

    def get_fbank(self, audio_or_path, stft_cfg, return_intermediate=False):
        r"""Helper function to get fbank from audio file."""
        if isinstance(audio_or_path, torch.Tensor):
            return maybe_add_dimension(audio_or_path, 3)
        
        if isinstance(audio_or_path, str):
            fbank, log_stft, wav = extract_fbank(
                audio_or_path,
                fn_STFT=self.fn_STFT,
                target_length=stft_cfg.filter_length,
                hop_size=stft_cfg.hop_length,
            )
            fbank = maybe_add_dimension(fbank, 3)  # (C,T,F)

            if return_intermediate:
                return fbank, log_stft, wav
            
            return fbank

    def wav2fbank(self, wav, target_length):
        fbank, log_magnitudes_stft = wav_to_fbank(wav, target_length, self.fn_STFT)
        return fbank, log_magnitudes_stft

    @torch.no_grad()
    def encode_fbank(self, fbank):
        latent = self.pipe.vae.encode(fbank)["latent_dist"].mean
        # NOTE: Scale the noise latent
        latent = latent * self.pipe.scheduler.init_noise_sigma
        return latent

    def ddim_inv(self, latent, prompt, emb_im=None, save_kv=True, mode="mix", prediction_type="epsilon"):
        ddim_inv = DDIMInversion(model=self.pipe, NUM_DDIM_STEPS=self.NUM_DDIM_STEPS)
        ddim_latents = ddim_inv.invert(
            ddim_latents=latent.unsqueeze(2), prompt=prompt, emb_im=emb_im,
            save_kv=save_kv, mode=mode, prediction_type=prediction_type
        )
        return ddim_latents
    
    def init_proj(self, precision):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to("cuda", dtype=precision)
        return image_proj_model

    # @torch.inference_mode()
    # def get_image_embeds(self, pil_image):
    #     if isinstance(pil_image, Image.Image):
    #         pil_image = [pil_image]
    #     clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
    #     clip_image = clip_image.to('cuda', dtype=self.precision)
    #     clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
    #     image_prompt_embeds = self.image_proj_model(clip_image_embeds)
    #     uncond_clip_image_embeds = self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[-2].detach()
    #     uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds).detach()
    #     return image_prompt_embeds, uncond_image_prompt_embeds

    def load_adapter(self):
        scale = 1.0
        attn_procs = {}
        for name in self.pipe.unet.attn_processors.keys():
            cross_attention_dim = None
            if name.startswith("mid_block"):
                hidden_size = self.pipe.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.pipe.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.pipe.unet.config.block_out_channels[block_id]
            # Only the first self-attention should be used for cross-attend different trojactory
            if name.endswith("attn1.processor"):
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=scale,
                    num_tokens=self.num_tokens,
                ).to("cuda", dtype=self.precision)
                
        self.pipe.unet.set_attn_processor(attn_procs)

    # def load_adapter(self, model_path, scale=1.0):
    #     from src.unet.attention_processor import IPAttnProcessor, AttnProcessor, Resampler
    #     attn_procs = {}
    #     for name in self.pipe.unet.attn_processors.keys():
    #         cross_attention_dim = None if name.endswith("attn1.processor") else self.pipe.unet.config.cross_attention_dim
    #         if name.startswith("mid_block"):
    #             hidden_size = self.pipe.unet.config.block_out_channels[-1]
    #         elif name.startswith("up_blocks"):
    #             block_id = int(name[len("up_blocks.")])
    #             hidden_size = list(reversed(self.pipe.unet.config.block_out_channels))[block_id]
    #         elif name.startswith("down_blocks"):
    #             block_id = int(name[len("down_blocks.")])
    #             hidden_size = self.pipe.unet.config.block_out_channels[block_id]
    #         if cross_attention_dim is None:
    #             attn_procs[name] = AttnProcessor()
    #         else:
    #             attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
    #             scale=scale,num_tokens= self.num_tokens).to('cuda', dtype=self.precision)
    #     self.pipe.unet.set_attn_processor(attn_procs)
    #     state_dict = torch.load(model_path, map_location="cpu")
    #     self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
    #     ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
    #     ip_layers.load_state_dict(state_dict["ip_adapter"], strict=True)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.pipe.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.pipe.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents with width->self.vocoder.config.model_in_dim
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            self.pipe.vocoder.config.model_in_dim // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.pipe.scheduler.init_noise_sigma
        return latents


if __name__ == "__main__":
    # pipeline = AudioLDMPipeline(
    #     sd_id="cvssp/audioldm-l-full", ip_id="cvssp/audioldm-l-full", NUM_DDIM_STEPS=50
    # )
    pipeline = TangoPipeline(
        sd_id="declare-lab/tango",
        ip_id="declare-lab/tango",
        NUM_DDIM_STEPS=50,
        precision=torch.float16,
    )
    print(pipeline.__dict__)