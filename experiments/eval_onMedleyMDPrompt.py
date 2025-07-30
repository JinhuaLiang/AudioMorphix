import os
import torch
import logging
import torchaudio
import numpy as np
import soundfile as sf

from src.audio_morphix import AudioMorphix
from src.utils.audio_processing import extract_fbank, maybe_add_dimension
from src.utils.factory import extract_and_fill
from src.utils.config import dynamic_config


# Setup logger
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s]%(name)s:\n%(message)s",
    handlers=[
        # logging.FileHandler('eval.log'),
        logging.StreamHandler()
    ],
)


# Setup Wandb
# wandb.login(key=os.environ['WANDB_API_KEY'])
# wandb_tags = [
#     f"guidance_scale: {cfgs.task.guidance_scale}",
#     f"energy_scale: {cfgs.task.energy_scale}",
#     f"w_edit: {cfgs.task.w_edit}",
#     f"w_content: {cfgs.task.w_content}",
#     f"sde_strength: {cfgs.task.sde_strength}",
# ]
# wandb_run = wandb.init(
#     project="Evaluation",
#     name=trial_name,
#     group=cfgs.wandb_group,
#     tags=wandb_tags,
#     mode='disabled' if cfgs.wandb_disable else 'online',
#     settings=wandb.Settings(_disable_stats=True),
#     job_type=cfgs.task.task,
#     config=OmegaConf.to_object(cfgs),
#     dir=output_dir,
#     )


n_sample_per_sec = 100  # 1024frames / 10.24s


def evaluate(model, cfgs):
    fbank_bg = model.editor.get_fbank(
        cfgs.task.background_audio_filepath,
        cfgs.audio_processor,
    )

    fbank_fg = model.editor.get_fbank(
        cfgs.task.foreground_audio_filepath,
        cfgs.audio_processor,
    )

    # Use part of ref if task is removal
    if cfgs.task.task in ["remove"]:
        fbank_fg = extract_and_fill(
            fbank_fg,
            stt_frame=cfgs.task.t_on,
            end_frame=cfgs.task.t_off,
            tgt_length=n_sample_per_sec,
        )

    mask_bg = torch.zeros_like(fbank_bg)
    mask_bg = mask_bg.squeeze()
    mask_bg[cfgs.task.t_on : cfgs.task.t_off, cfgs.task.f_low : cfgs.task.f_up] = 1
    # Create a mask fg without scaling op
    mask_fg = torch.roll(mask_bg, (int(cfgs.task.dt), int(cfgs.task.df)), (-2, -1))

    if cfgs.task.task == "paste":
        result = model.run_paste(
            fbank_bg=fbank_bg,
            mask_bg=mask_bg,
            fbank_fg=fbank_fg,
            prompt=cfgs.task.background_audio_caption,
            prompt_replace=cfgs.task.foreground_audio_caption,
            dx=cfgs.task.df,
            dy=cfgs.task.dt,
            resize_scale=cfgs.task.resize_scale,
            guidance_scale=cfgs.task.guidance_scale,
            energy_scale=cfgs.task.energy_scale,
            w_edit=cfgs.task.w_edit,
            w_content=cfgs.task.w_content,
            SDE_strength=cfgs.task.sde_strength,
            seed=cfgs.seed,
            save_kv=True,
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
            resize_scale=cfgs.task.resize_scale,
            guidance_scale=cfgs.task.guidance_scale,
            energy_scale=cfgs.task.energy_scale,
            w_edit=cfgs.task.w_edit,
            w_content=cfgs.task.w_content,
            SDE_strength=cfgs.task.sde_strength,
            seed=cfgs.seed,
            save_kv=True,
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
            resize_scale=cfgs.task.resize_scale,
            guidance_scale=cfgs.task.guidance_scale,
            energy_scale=cfgs.task.energy_scale,
            w_edit=cfgs.task.w_edit,
            w_contrast=cfgs.task.w_contrast,
            w_content=cfgs.task.w_content,
            SDE_strength=cfgs.task.sde_strength,
            seed=cfgs.seed,
            save_kv=True,
            bg_to_fg_ratio=0.5,
            iterations=50,
            enable_penalty=True,
            disable_tangent_proj=cfgs.task.disable_tangent_proj,
        )
    elif cfgs.task.task == "replace":
        result = model.run_remove(
            fbank_bg=fbank_bg,
            mask_bg=mask_bg,
            fbank_fg=fbank_fg,
            prompt=cfgs.task.background_audio_caption,
            prompt_replace=cfgs.task.foreground_audio_caption,
            dx=cfgs.task.df,
            dy=cfgs.task.dt,
            resize_scale=cfgs.task.resize_scale,
            guidance_scale=cfgs.task.guidance_scale,
            energy_scale=cfgs.task.energy_scale,
            w_edit=cfgs.task.w_edit[0],
            w_contrast=cfgs.task.w_contrast,
            w_content=cfgs.task.w_content,
            SDE_strength=cfgs.task.sde_strength,
            seed=cfgs.seed,
            save_kv=True,
            bg_to_fg_ratio=0.5,
            iterations=50,
            enable_penalty=True,
            disable_tangent_proj=cfgs.task.disable_tangent_proj,
        )
        # Obtain target reference audio
        fbank_ref, log_stft_ref, wav_ref = extract_fbank(
            cfgs.task.reference_audio_filepath,
            fn_STFT=fn_stft,
            target_length=cfgs.audio_processor.filter_length,
            hop_size=cfgs.audio_processor.hop_length,
        )
        fbank_ref = maybe_add_dimension(fbank_ref, 3)

        log.warning(cfgs.task.w_edit[1])
        result = model.run_mix(
            fbank_bg=result.mel_spectrogram,
            mask_bg=mask_bg,
            fbank_fg=fbank_ref,
            prompt=cfgs.task.context,
            prompt_replace=cfgs.task.reference_audio_caption,
            dx=cfgs.task.df,
            dy=cfgs.task.dt,
            resize_scale=cfgs.task.resize_scale,
            guidance_scale=cfgs.task.guidance_scale,
            energy_scale=cfgs.task.energy_scale,
            w_edit=cfgs.task.w_edit[1],
            w_content=cfgs.task.w_content,
            SDE_strength=cfgs.task.sde_strength,
            seed=cfgs.seed,
            save_kv=True,
            bg_to_fg_ratio=0.7,
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
            resize_scale=cfgs.task.resize_scale,
            guidance_scale=cfgs.task.guidance_scale,
            energy_scale=cfgs.task.energy_scale,
            w_edit=cfgs.task.w_edit,
            w_content=cfgs.task.w_content,
            SDE_strength=0,
            seed=cfgs.seed,
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
            resize_scale=cfgs.task.resize_scale,
            guidance_scale=cfgs.task.guidance_scale,
            energy_scale=cfgs.task.energy_scale,
            w_edit=cfgs.task.w_edit,
            w_content=cfgs.task.w_content,
            SDE_strength=0,
            seed=cfgs.seed,
            save_kv=True,
            disable_tangent_proj=cfgs.task.disable_tangent_proj,
        )
    else:
        raise ValueError(f"Cannot run the task {cfgs.task.task}")

    # wandb.log({
    #     'background_wav': wandb.Audio(wav_bg.cpu().squeeze().numpy(), caption=cfgs.task.background_audio_caption, sample_rate=cfgs.audio_processor.sampling_rate),
    #     'background_spec': wandb.Image(plot_spectrogram(fbank_bg.permute(0,2,1)), caption=cfgs.task.background_audio_caption),
    #     'background_mask': wandb.Image(plot_spectrogram(mask_bg.permute(1,0)), caption="Mask of background sound"),
    #     'foreground_wav': wandb.Audio(wav_fg.cpu().squeeze().numpy(), caption=cfgs.task.foreground_audio_caption, sample_rate=cfgs.audio_processor.sampling_rate),
    #     'foreground_spec': wandb.Image(plot_spectrogram(fbank_fg.permute(0,2,1)), caption=cfgs.task.foreground_audio_caption),
    #     'foreground_mask': wandb.Image(plot_spectrogram(mask_fg.permute(1,0)), caption="Mask of foreground sound"),
    #     'generated_wav': wandb.Audio(result.waveform.cpu().squeeze().numpy(), caption="After adding foreground sound.", sample_rate=cfgs.audio_processor.sampling_rate),
    #     'generated_spec': wandb.Image(plot_spectrogram(result.mel_spectrogram.permute(0,1,3,2)), caption="After adding foreground sound."),
    # })

    edited_wav = result.waveform

    if cfgs.task.output_audio_filepath is not None:
        if isinstance(edited_wav, np.ndarray):
            sf.write(
                cfgs.task.output_audio_filepath,
                edited_wav[0],
                samplerate=cfgs.audio_processor.sampling_rate,
            )
        else:
            torchaudio.save(
                cfgs.task.output_audio_filepath,
                edited_wav,
                sample_rate=cfgs.audio_processor.sampling_rate,
            )


if __name__ == "__main__":
    from tqdm import tqdm

    from src.utils.factory import load_json

    # Config and setup a trial
    cfgs = dynamic_config("Evaluate Audio Morph.")
    trial_name = f"{cfgs.tag}"
    output_dir = os.path.join(cfgs.output_dir, trial_name)
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Set the output of the trial to {output_dir}")

    tgt_audio_dir = os.path.join(output_dir, "target")
    out_audio_dir = os.path.join(output_dir, "generation")
    os.makedirs(tgt_audio_dir, exist_ok=True)
    os.makedirs(out_audio_dir, exist_ok=True)

    dataset = load_json(cfgs.json_path)

    # Build up model
    model = AudioMorphix(
        pretrained_model_path=cfgs.model,
        num_ddim_steps=cfgs.task.num_ddim_steps,
    )

    with tqdm(total=10 if cfgs.mini_data else len(dataset)) as pbar:
        for i, datum in enumerate(dataset):
            if i == 10 and cfgs.mini_data:
                break

            pbar.update(1)

            # Resume inference if applicable
            tgt_audio_path = os.path.join(tgt_audio_dir, f"{i}.wav")
            out_audio_path = os.path.join(out_audio_dir, f"{i}.wav")
            if not cfgs.resume and os.path.exists(out_audio_path):
                continue

            cfgs.task.background_audio_filepath = datum["file_path"]
            cfgs.task.background_audio_caption = datum["source_caption"]
            cfgs.task.foreground_audio_filepath = datum["file_path"]
            cfgs.task.foreground_audio_caption = datum["target_caption"]
            cfgs.task.output_audio_filepath = out_audio_path

            os.system(f"cp {datum['file_path']} {tgt_audio_path}")

            evaluate(model, cfgs)
