import os
import torch
import torchaudio
import logging
from diffusers import AudioLDMPipeline
from pytorch_lightning import seed_everything

from src.data import label2caption
from src.audio_morphix import SoundEditorOutput
from src.utils.inversion import DDIMInversion
from src.utils.audio_processing import TacotronSTFT, extract_fbank, maybe_add_dimension


log = logging.getLogger(__name__)

# NOTE: stft is fixed due to the vocoder used in the audioldm
audio_cfgs = {
    "filter_length": 1024,
    "hop_length": 160,
    "win_length": 1024,
    "n_mel_channels": 64,
    "sampling_rate": 16000,
    "mel_fmin": 0,
    "mel_fmax": 8000,
}


def get_fbank_from_file(file_path):
    stft_fn = TacotronSTFT(**audio_cfgs)
    fbank, log_magnitudes_stft, waveform = extract_fbank(
        file_path, fn_STFT=stft_fn, target_length=1024, hop_size=160
    )  # fix length
    return fbank, log_magnitudes_stft, waveform


def fbank2latent(fbank, pipeline):
    latent = pipeline.vae.encode(fbank)["latent_dist"].mean
    # NOTE: Scale the noise latent
    latent = latent * pipeline.scheduler.init_noise_sigma
    return latent


def run_ddim_inversion(
    file_path,
    prompt_src,
    prompt_tgt,
    n_ddim_steps=50,
    repo_id="cvssp/audioldm",
    seed=42,
):
    seed_everything(seed)

    fbank_src, _, _ = get_fbank_from_file(file_path)
    # Prepare input spec with shape = (B,C,T,F)
    fbank_src = maybe_add_dimension(fbank_src, 4).to("cuda", dtype=torch.float16)

    pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16).to(
        "cuda"
    )
    latent_src = fbank2latent(fbank_src, pipe)

    ddim_inv = DDIMInversion(model=pipe, NUM_DDIM_STEPS=n_ddim_steps)
    ddim_latents = ddim_inv.invert(
        ddim_latents=torch.cat([latent_src, latent_src]),
        prompt=[prompt_src, prompt_src],
        emb_im=None,
    )
    latent_in = ddim_latents[-1][:1].squeeze(2)

    # Generate new sound using new prompt
    audio = pipe(
        prompt_tgt,
        latents=latent_in,
        num_inference_steps=n_ddim_steps,
        audio_length_in_s=10.0,
    ).audios[0]

    return torch.from_numpy(audio).unsqueeze(0)


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    from engine import load_json

    # I/O
    parser = argparse.ArgumentParser("Inference by using DDIM Inversion.")
    parser.add_argument("--task", type=str)
    parser.add_argument("--json-path", type=str)
    parser.add_argument("--output-dir", type=str, default="./output/ddim_inv")
    parser.add_argument("--mini-data", action="store_true", default=False)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Set the output of the trial to {output_dir}")

    tgt_audio_dir = os.path.join(args.output_dir, "target")
    out_audio_dir = os.path.join(args.output_dir, "generation")
    os.makedirs(tgt_audio_dir, exist_ok=True)
    os.makedirs(out_audio_dir, exist_ok=True)

    # Begin to inference
    dataset = load_json(args.json_path)

    with tqdm(total=5 if args.mini_data else len(dataset)) as pbar:
        for i, datum in enumerate(dataset):
            if i == 5 and args.mini_data:
                break

            pbar.update(1)
            # if i % 4 != 0:
            #     continue

            # Resume inference if applicable
            tgt_audio_path = os.path.join(tgt_audio_dir, f"{i}.wav")
            out_audio_path = os.path.join(out_audio_dir, f"{i}.wav")
            if os.path.exists(out_audio_path):
                continue

            os.system(f"cp {datum['file_path']} {tgt_audio_path}")
            out_audio = run_ddim_inversion(
                file_path=datum["file_path"],
                prompt_src=datum["source_caption"],
                prompt_tgt=datum["target_caption"],
            )
            torchaudio.save(
                out_audio_path, out_audio, sample_rate=audio_cfgs["sampling_rate"]
            )
