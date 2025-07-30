import os
import torch
import wandb
import shutil
import logging
import numpy as np
from omegaconf import OmegaConf

from src.audio_morphix import AudioMorphix
from src.utils.factory import (
    plot_spectrogram,
    get_current_time,
    get_edit_mask,
)
from src.utils.config import dynamic_config, dump_config


n_sample_per_sec = 100  # 1024frames / 10.24s


# Setup logger
log = logging.getLogger(__name__)

# Config and setup a trial
cfgs = dynamic_config("A demo of AudioMorphix.")
current_time = get_current_time()
trial_name = f"{cfgs.wandb_name}-{current_time}"
output_dir = os.path.join(cfgs.output_dir, trial_name)
os.makedirs(output_dir, exist_ok=True)
shutil.copyfile("scripts/run.sh", os.path.join(output_dir, "trial.sh"))
dump_config(cfgs, dump_path=os.path.join(output_dir, "trial_config.yaml"))
log.info(f"Set the output of the trial to {output_dir}")

# Setup Wandb
wandb.login(key=os.environ["WANDB_API_KEY"])
wandb_tags = [
    f"guidance_scale: {cfgs.task.guidance_scale}",
    f"energy_scale: {cfgs.task.energy_scale}",
    f"w_edit: {cfgs.task.w_edit}",
    f"w_content: {cfgs.task.w_content}",
    f"sde_strength: {cfgs.task.sde_strength}",
]
wandb_run = wandb.init(
    project="eval_AudioMorphix",
    name=trial_name,
    group=cfgs.wandb_group,
    tags=wandb_tags,
    mode="disabled" if cfgs.wandb_disable else "online",
    settings=wandb.Settings(_disable_stats=True),
    job_type=cfgs.task.task,
    config=OmegaConf.to_object(cfgs),
    dir=output_dir,
)


model = AudioMorphix(
    pretrained_model_path=cfgs.model,
    num_ddim_steps=cfgs.task.num_ddim_steps,
    )

fbank_bg, log_stft_bg, wav_bg = model.editor.get_fbank(
    cfgs.task.background_audio_filepath, 
    cfgs.audio_processor,
    return_intermediate=True,
    )
    
fbank_fg, log_stft_fg, wav_fg = model.editor.get_fbank(
    cfgs.task.foreground_audio_filepath, 
    cfgs.audio_processor,
    return_intermediate=True,
    )

fbank_fg_ori = fbank_fg

mask_bg = torch.zeros_like(fbank_bg)
mask_bg = mask_bg.squeeze()
mask_bg[cfgs.task.t_on : cfgs.task.t_off, cfgs.task.f_low : cfgs.task.f_up] = 1

# Create a mask fg with scaling operation
mask_fg = get_edit_mask(
    mask_bg, dx=cfgs.task.df, dy=cfgs.task.dt, 
    resize_scale_x=cfgs.task.resize_scale_f, resize_scale_y=cfgs.task.resize_scale_t,
)

if cfgs.task.task == "paste":
    result = model.run_paste(
        fbank_bg=fbank_bg,
        mask_bg=mask_bg,
        fbank_fg=fbank_fg,
        prompt=cfgs.task.background_audio_caption,
        prompt_replace=cfgs.task.foreground_audio_caption,
        dx=cfgs.task.df,
        dy=cfgs.task.dt,
        resize_scale_x=cfgs.task.resize_scale_f,
        resize_scale_y=cfgs.task.resize_scale_t,
        guidance_scale=cfgs.task.guidance_scale,
        energy_scale=cfgs.task.energy_scale,
        w_edit=cfgs.task.w_edit,
        w_content=cfgs.task.w_content,  # 5
        SDE_strength=cfgs.task.sde_strength,
        seed=cfgs.seed,
        save_kv=(not cfgs.task.disable_kv_cache), # True,
        disable_tangent_proj=cfgs.task.disable_tangent_proj,
    )
elif cfgs.task.task == "mix":
    result = model.run_mix(
        fbank_bg=fbank_bg,
        mask_bg=mask_bg,
        fbank_fg=fbank_fg,
        prompt=cfgs.task.background_audio_caption,
        prompt_replace=cfgs.task.foreground_audio_caption,
        dx=cfgs.task.df,
        dy=cfgs.task.dt,
        resize_scale_x=cfgs.task.resize_scale_f,
        resize_scale_y=cfgs.task.resize_scale_t,
        guidance_scale=cfgs.task.guidance_scale,
        energy_scale=cfgs.task.energy_scale,
        w_edit=cfgs.task.w_edit,
        w_content=cfgs.task.w_content,
        SDE_strength=cfgs.task.sde_strength,
        seed=cfgs.seed,
        save_kv=(not cfgs.task.disable_kv_cache), # True,
        disable_tangent_proj=cfgs.task.disable_tangent_proj,
    )
elif cfgs.task.task == "remove":
    result = model.run_remove(
        fbank_bg=fbank_bg,
        mask_bg=mask_bg,
        fbank_fg=fbank_fg,
        prompt=cfgs.task.background_audio_caption,
        prompt_replace=cfgs.task.foreground_audio_caption,
        dx=cfgs.task.df,
        dy=cfgs.task.dt,
        resize_scale_x=cfgs.task.resize_scale_f,
        resize_scale_y=cfgs.task.resize_scale_t,
        guidance_scale=cfgs.task.guidance_scale,
        energy_scale=cfgs.task.energy_scale,
        w_edit=cfgs.task.w_edit,
        w_contrast=cfgs.task.w_contrast,
        w_content=cfgs.task.w_content,
        SDE_strength=cfgs.task.sde_strength,
        seed=cfgs.seed,
        save_kv=(not cfgs.task.disable_kv_cache), # True,
        bg_to_fg_ratio=0.5,
        iterations=50,
        enable_penalty=True,
        disable_tangent_proj=cfgs.task.disable_tangent_proj,
    )
elif cfgs.task.task == "generate":
    result = model.run_audio_generation(
        fbank_bg=fbank_bg,
        mask_bg=mask_bg,
        fbank_fg=fbank_fg,
        prompt=cfgs.task.background_audio_caption,
        prompt_replace=cfgs.task.foreground_audio_caption,
        dx=cfgs.task.df,
        dy=cfgs.task.dt,
        resize_scale_x=cfgs.task.resize_scale_f,
        resize_scale_y=cfgs.task.resize_scale_t,
        guidance_scale=cfgs.task.guidance_scale,
        energy_scale=cfgs.task.energy_scale,
        w_edit=cfgs.task.w_edit,
        w_content=cfgs.task.w_content,
        SDE_strength=0,
        seed=cfgs.seed,
        save_kv=(not cfgs.task.disable_kv_cache), # True,
        disable_tangent_proj=cfgs.task.disable_tangent_proj,
    )
elif cfgs.task.task == "style_transfer":
    result = model.run_style_transferring(
        fbank_bg=fbank_bg,
        mask_bg=mask_bg,
        fbank_fg=fbank_fg,
        prompt=cfgs.task.background_audio_caption,
        prompt_replace=cfgs.task.foreground_audio_caption,
        dx=cfgs.task.df,
        dy=cfgs.task.dt,
        resize_scale_x=cfgs.task.resize_scale_f,
        resize_scale_y=cfgs.task.resize_scale_t,
        guidance_scale=cfgs.task.guidance_scale,
        energy_scale=cfgs.task.energy_scale,
        w_edit=cfgs.task.w_edit,
        w_content=cfgs.task.w_content,
        SDE_strength=0,
        seed=cfgs.seed,
        save_kv=(not cfgs.task.disable_kv_cache), # True,
        disable_tangent_proj=cfgs.task.disable_tangent_proj,
    )
elif cfgs.task.task == "move_and_resize":
    if (cfgs.task.t_on_keep is None) or (cfgs.task.t_off_keep is None) or (cfgs.task.f_low_keep is None) or (cfgs.task.f_up_keep is None):
        mask_keep = None
    else:
        mask_keep = torch.zeros_like(fbank_bg)
        mask_keep = mask_keep.squeeze()
        mask_keep[int(cfgs.task.t_on_keep):int(cfgs.task.t_off_keep), int(cfgs.task.f_low_keep):int(cfgs.task.f_up_keep)] = 1

    result = model.run_move(
        fbank_org=fbank_bg,
        mask=mask_bg,
        dx=cfgs.task.df,
        dy=cfgs.task.dt,
        mask_ref=None,
        mask_keep=mask_keep,
        prompt=cfgs.task.background_audio_caption,
        resize_scale_x=cfgs.task.resize_scale_f,
        resize_scale_y=cfgs.task.resize_scale_t,
        guidance_scale=cfgs.task.guidance_scale,
        energy_scale=cfgs.task.energy_scale,
        w_edit=cfgs.task.w_edit,
        w_content=cfgs.task.w_content,
        w_contrast=cfgs.task.w_contrast,
        w_inpaint=cfgs.task.w_inpaint,
        SDE_strength=cfgs.task.sde_strength,
        seed=cfgs.seed,
        save_kv=(not cfgs.task.disable_kv_cache),
        disable_tangent_proj=cfgs.task.disable_tangent_proj,
        scale_denoised=True,
    )
else:
    raise ValueError(f"Cannot run the task {cfgs.task.task}")


edited_wav = result.waveform

if isinstance(edited_wav, np.ndarray):  # tango
    fbank_bg, fbank_fg_ori = fbank_bg[0], fbank_fg_ori[0]
    edited_wav = edited_wav[0]
else:
    edited_wav = edited_wav.cpu().squeeze().numpy()


wandb.log(
    {
        "background_wav": wandb.Audio(
            wav_bg.cpu().squeeze().numpy(),
            caption=cfgs.task.background_audio_caption,
            sample_rate=cfgs.audio_processor.sampling_rate,
        ),
        "background_spec": wandb.Image(
            plot_spectrogram(fbank_bg.permute(0, 2, 1)[:,:,:10*n_sample_per_sec]), # discard padding area
            caption=cfgs.task.background_audio_caption,
        ),
        "background_mask": wandb.Image(
            plot_spectrogram(mask_bg.permute(1, 0)[:,:10*n_sample_per_sec], auto_amp=True), caption="Mask of background sound"
        ),
        "foreground_wav": wandb.Audio(
            wav_fg.cpu().squeeze().numpy(),
            caption=cfgs.task.foreground_audio_caption,
            sample_rate=cfgs.audio_processor.sampling_rate,
        ),
        "foreground_spec": wandb.Image(
            plot_spectrogram(fbank_fg_ori.permute(0, 2, 1)[:,:,:10*n_sample_per_sec], filename="out.png"),
            caption=cfgs.task.foreground_audio_caption,
        ),
        "foreground_mask": wandb.Image(
            plot_spectrogram(mask_fg.permute(1, 0)[:,:10*n_sample_per_sec], auto_amp=True), caption="Mask of foreground sound"
        ),
        "generated_wav": wandb.Audio(
            edited_wav,
            caption="After adding foreground sound.",
            sample_rate=cfgs.audio_processor.sampling_rate,
        ),
        "generated_spec": wandb.Image(
            plot_spectrogram(result.mel_spectrogram.permute(0, 1, 3, 2)[:,:,:,:10*n_sample_per_sec]),
            caption="After adding foreground sound.",
        ),
    }
)


# if cfgs.task.output_audio_filepath is not None:
#     if isinstance(edited_wav, np.ndarray):
#         sf.write(cfgs.task.output_audio_filepath, edited_wav[0], samplerate=cfgs.audio_processor.sampling_rate,)
#     else:
#         torchaudio.save(
#             cfgs.task.output_audio_filepath,
#             edited_wav,
#             sample_rate=cfgs.audio_processor.sampling_rate,
#         )
