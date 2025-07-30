import torch
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass
from pytorch_lightning import seed_everything


from src.model.pipeline import AudioLDMPipeline, TangoPipeline
from src.utils.utils import (
    process_move,
    process_paste,
    process_remove,
)
from src.utils.audio_processing import TacotronSTFT, wav_to_fbank, maybe_add_dimension
from src.utils.factory import slerp, fill_with_neighbor, optimize_neighborhood_points


# NUM_DDIM_STEPS = 50 # 50
SIZES = {
    0: 4,
    1: 2,
    2: 1,
    3: 1,
}


@dataclass
class SoundEditorOutput:
    waveform: torch.tensor
    mel_spectrogram: torch.tensor


class AudioMorphix:
    def __init__(
            self, 
            pretrained_model_path, 
            num_ddim_steps=50,
            device = "cuda" if torch.cuda.is_available() else "cpu",
            ):
        self.ip_scale = 0.1
        self.precision = torch.float32  # torch.float16
        if "audioldm" in pretrained_model_path:
            _pipe_cls = AudioLDMPipeline
        elif "tango" in pretrained_model_path:
            _pipe_cls = TangoPipeline
        self.editor = _pipe_cls(
            sd_id=pretrained_model_path,
            NUM_DDIM_STEPS=num_ddim_steps,
            precision=self.precision,
            ip_scale=self.ip_scale,
            device=device,
        )

        self.up_ft_index = [2, 3]  # fixed in gradio demo  # TODO: change to 2,3
        self.up_scale = 2  # fixed in gradio demo
        self.device = device
        self.num_ddim_steps = num_ddim_steps


    def to(self, device):
        self.editor.pipe = self.editor.pipe.to(device)
        self.editor.pipe._device = device
        self.editor.device = device
        self.device = device


    def run_move(
        self, 
        fbank_org, 
        mask, 
        dx, dy, 
        mask_ref, 
        prompt, 
        resize_scale_x, 
        resize_scale_y, 
        w_edit, 
        w_content, 
        w_contrast, 
        w_inpaint, 
        seed, 
        guidance_scale, 
        energy_scale, 
        SDE_strength, 
        mask_keep=None, 
        ip_scale=None, 
        save_kv=False, 
        disable_tangent_proj=False, 
        scale_denoised=True,
        ):
        seed_everything(seed)
        energy_scale = energy_scale * 1e3

        # Prepare input spec and mask
        input_scale = 1
        fbank_org = maybe_add_dimension(fbank_org, 4).to(
            self.device, dtype=self.precision
        )  # shape = (B,C,T,F)
        f, t = fbank_org.shape[-1], fbank_org.shape[-2]

        if save_kv:
            self.editor.load_adapter()

        ### FIXME
        if mask_ref is not None and np.sum(mask_ref) != 0:
            mask_ref = np.repeat(mask_ref[:,:,None], 3, 2)
        else:
            mask_ref = None

        latent = self.editor.fbank2latent(fbank_org)
        ddim_latents = self.editor.ddim_inv(latent=latent, prompt=prompt)
        latent_in = ddim_latents[-1].squeeze(2)

        scale = 4 * SIZES[max(self.up_ft_index)] / self.up_scale

        edit_kwargs = process_move(
            path_mask=mask,
            h=f,
            w=t,
            dx=dx,
            dy=dy,
            scale=scale,
            input_scale=input_scale,
            resize_scale_x=resize_scale_x,
            resize_scale_y=resize_scale_y,
            up_scale=self.up_scale,
            up_ft_index=self.up_ft_index,
            w_edit=w_edit,
            w_content=w_content,
            w_contrast=w_contrast,
            w_inpaint=w_inpaint,
            precision=self.precision,
            path_mask_ref=mask_ref,
            path_mask_keep=mask_keep,
        )
        # Pre-process zT
        mask_tmp = (F.interpolate(mask.unsqueeze(0).unsqueeze(0), (int(latent_in.shape[-2]*resize_scale_y), int(latent_in.shape[-1]*resize_scale_x)))>0).float().to('cuda', dtype=latent_in.dtype)
        latent_tmp = F.interpolate(latent_in, (int(latent_in.shape[-2]*resize_scale_y), int(latent_in.shape[-1]*resize_scale_x)))
        mask_tmp = torch.roll(mask_tmp, (int(dy/(t/latent_in.shape[-2])*resize_scale_y), int(dx/(t/latent_in.shape[-2])*resize_scale_x)), (-2,-1))
        latent_tmp = torch.roll(latent_tmp, (int(dy/(t/latent_in.shape[-2])*resize_scale_y), int(dx/(t/latent_in.shape[-2])*resize_scale_x)), (-2,-1))

        _mask_temp = torch.zeros(1,1,latent_in.shape[-2], latent_in.shape[-1]).to(
            latent_in.device, dtype=latent_in.dtype)
        _latent_temp = torch.zeros_like(latent_in)
        
        pad_x = (_mask_temp.shape[-1] - mask_tmp.shape[-1]) // 2
        pad_y = (_mask_temp.shape[-2] - mask_tmp.shape[-2]) // 2
        px_tmp, py_tmp = max(pad_x, 0), max(pad_y, 0)
        px_tar, py_tar = max(-pad_x, 0), max(-pad_y, 0)
        _mask_temp[:,:,py_tmp:mask_tmp.shape[-2]+py_tmp,px_tmp:mask_tmp.shape[-1]+px_tmp] = mask_tmp[
            :,:,py_tar:_mask_temp.shape[-2]+py_tar,px_tar:_mask_temp.shape[-1]+px_tar]
        _latent_temp[:,:,py_tmp:latent_tmp.shape[-2]+py_tmp,px_tmp:latent_tmp.shape[-1]+px_tmp] = latent_tmp[
            :,:,py_tar:_latent_temp.shape[-2]+py_tar,px_tar:_latent_temp.shape[-1]+px_tar]

        mask_tmp = (_mask_temp>0.5).float()
        latent_tmp = _latent_temp

        if edit_kwargs["mask_keep"] is not None:
            mask_keep = edit_kwargs["mask_keep"]
            mask_keep = (F.interpolate(mask_keep, (latent_in.shape[-2], latent_in.shape[-1]))>0).float().to('cuda', dtype=latent_in.dtype)
        else:
            mask_keep = 1 - mask_tmp

        latent_in = (torch.zeros_like(latent_in)+latent_in*mask_keep+latent_tmp*mask_tmp).to(dtype=latent_in.dtype)

        latent_rec = self.editor.pipe.edit(
            mode='move',
            latent=latent_in,
            prompt=prompt,
            guidance_scale=guidance_scale,
            energy_scale=energy_scale,
            latent_noise_ref=ddim_latents,
            SDE_strength=SDE_strength,
            edit_kwargs=edit_kwargs,
            disable_tangent_proj=disable_tangent_proj,
        )

        # Scale output latent
        if scale_denoised:
            _max = torch.max(torch.abs(latent_rec))
            latent_rec = latent_rec * 5 / _max

        spec_rec = self.editor.decode_latents(latent_rec)
        wav_rc = self.editor.mel_spectrogram_to_waveform(spec_rec)

        torch.cuda.empty_cache()

        return SoundEditorOutput(wav_rc, spec_rec)


    def run_paste(
        self,
        fbank_bg,
        mask_bg,
        fbank_fg,
        prompt,
        prompt_replace,
        w_edit,
        w_content,
        seed,
        guidance_scale,
        energy_scale,
        dx,
        dy,
        resize_scale_x, 
        resize_scale_y,
        SDE_strength,
        save_kv=False,
        disable_tangent_proj=False,
        scale_denoised=True,
    ):
        seed_everything(seed)
        energy_scale = energy_scale * 1e3

        # Prepare input spec and mask
        input_scale = 1
        fbank_bg = maybe_add_dimension(fbank_bg, 4).to(
            self.device, dtype=self.precision
        )  # shape = (B,C,T,F)
        f, t = fbank_bg.shape[-1], fbank_bg.shape[-2]
        fbank_fg = maybe_add_dimension(fbank_fg, 4).to(
            self.device, dtype=self.precision
        )

        # mask_bg = maybe_add_dimension(mask_bg, 3).permute(1,2,0).numpy().astype('uint8') # shape = (C,T,F)
        # mask_bg = mask_bg.numpy().astype('uint8') # shape = (C,T,F)

        if save_kv:
            self.editor.load_adapter()

        latent_base = self.editor.fbank2latent(fbank_bg)

        #####[START] Original rescale and fit method.#####
        # if resize_scale != 1:
        #     hr, wr = fbank_fg.shape[-2], fbank_fg.shape[-1]
        #     fbank_fg = F.interpolate(
        #         fbank_fg, (int(hr * resize_scale), int(wr * resize_scale))
        #     )
        #     pad_size_x = abs(fbank_fg.shape[-1] - wr) // 2
        #     pad_size_y = abs(fbank_fg.shape[-2] - hr) // 2
        #     if resize_scale > 1:
        #         fbank_fg = fbank_fg[
        #             :, :, pad_size_y : pad_size_y + hr, pad_size_x : pad_size_x + wr
        #         ]
        #     else:
        #         temp = torch.zeros(1, 1, hr, wr).to(self.device, dtype=self.precision)
        #         temp[
        #             :,
        #             :,
        #             pad_size_y : pad_size_y + fbank_fg.shape[-2],
        #             pad_size_x : pad_size_x + fbank_fg.shape[-1],
        #         ] = fbank_fg
        #         fbank_fg = temp
        #####[END] Original rescale and fit method.#####
        hr, wr = fbank_fg.shape[-2], fbank_fg.shape[-1]
        fbank_tmp = torch.zeros_like(fbank_fg)
        fbank_fg = F.interpolate(
            fbank_fg, (int(hr * resize_scale_y), int(wr * resize_scale_x))
            )
        pad_x = (wr - fbank_fg.shape[-1]) // 2
        pad_y = (hr - fbank_fg.shape[-2]) // 2
        px_tmp, py_tmp = max(pad_x, 0), max(pad_y, 0)
        px_tar, py_tar = max(-pad_x, 0), max(-pad_y, 0)

        fbank_tmp[:,:,py_tmp:fbank_fg.shape[-2]+py_tmp,px_tmp:fbank_fg.shape[-1]+px_tmp] = fbank_fg[
            :,:,py_tar:fbank_tmp.shape[-2]+py_tar,px_tar:fbank_tmp.shape[-1]+px_tar]
        fbank_fg = fbank_tmp

        latent_replace = self.editor.fbank2latent(fbank_fg)
        ddim_latents = self.editor.ddim_inv(
            latent=torch.cat([latent_base, latent_replace]),
            prompt=[prompt, prompt_replace],
        )

        latent_in = ddim_latents[-1][:1].squeeze(2)  # latent_base_noise

        scale = 8 * SIZES[max(self.up_ft_index)] / self.up_scale / 2

        edit_kwargs = process_paste(
            path_mask=mask_bg,
            h=f,
            w=t,
            dx=dx,
            dy=dy,
            scale=scale,
            input_scale=input_scale,
            up_scale=self.up_scale,
            up_ft_index=self.up_ft_index,
            w_edit=w_edit,
            w_content=w_content,
            precision=self.precision,
            resize_scale_x=resize_scale_x,
            resize_scale_y=resize_scale_y,
        )
        mask_tmp = (
            F.interpolate(
                edit_kwargs["mask_base_cur"].float(),
                (latent_in.shape[-2], latent_in.shape[-1]),
            )
            > 0
        ).float()
        # latent_replace_noise with rolling
        latent_tmp = torch.roll(
            ddim_latents[-1][1:].squeeze(2),
            (int(dy / (t / latent_in.shape[-2])), int(dx / (t / latent_in.shape[-2]))),
            (-2, -1),
        )
        # blended latent
        latent_in = (latent_in * (1 - mask_tmp) + latent_tmp * mask_tmp).to(
            dtype=latent_in.dtype
        )

        latent_rec = self.editor.pipe.edit(
            mode="paste",
            latent=latent_in,
            prompt=prompt,
            guidance_scale=guidance_scale,
            energy_scale=energy_scale,
            latent_noise_ref=ddim_latents,
            SDE_strength=SDE_strength,
            edit_kwargs=edit_kwargs,
            disable_tangent_proj=disable_tangent_proj,
        )

        # Scale output latent
        if scale_denoised:
            _max = torch.max(torch.abs(latent_rec))
            latent_rec = latent_rec * 5 / _max

        spec_rec = self.editor.decode_latents(latent_rec)
        wav_rc = self.editor.mel_spectrogram_to_waveform(spec_rec)
        torch.cuda.empty_cache()

        return SoundEditorOutput(wav_rc, spec_rec)

    def run_mix(
        self,
        fbank_bg,
        mask_bg,
        fbank_fg,
        prompt,
        prompt_replace,
        w_edit,
        w_content,
        seed,
        guidance_scale,
        energy_scale,
        dx,
        dy,
        resize_scale_x,
        resize_scale_y,
        SDE_strength,
        save_kv=False,
        bg_to_fg_ratio=0.7,
        disable_tangent_proj=False,
        scale_denoised=False,
    ):
        seed_everything(seed)
        energy_scale = energy_scale * 1e3

        # Prepare input spec and mask
        input_scale = 1
        fbank_bg = maybe_add_dimension(fbank_bg, 4).to(
            self.device, dtype=self.precision
        )  # shape = (B,C,T,F)
        f, t = fbank_bg.shape[-1], fbank_bg.shape[-2]
        fbank_fg = maybe_add_dimension(fbank_fg, 4).to(
            self.device, dtype=self.precision
        )

        if save_kv:
            self.editor.load_adapter()

        latent_base = self.editor.fbank2latent(fbank_bg)

        #####[START] Original rescale and fit method.#####
        # if resize_scale != 1:
        #     hr, wr = fbank_fg.shape[-2], fbank_fg.shape[-1]
        #     fbank_fg = F.interpolate(
        #         fbank_fg, (int(hr * resize_scale), int(wr * resize_scale))
        #     )
        #     pad_size_x = abs(fbank_fg.shape[-1] - wr) // 2
        #     pad_size_y = abs(fbank_fg.shape[-2] - hr) // 2
        #     if resize_scale > 1:
        #         fbank_fg = fbank_fg[
        #             :, :, pad_size_y : pad_size_y + hr, pad_size_x : pad_size_x + wr
        #         ]
        #     else:
        #         temp = torch.zeros(1, 1, hr, wr).to(self.device, dtype=self.precision)
        #         temp[
        #             :,
        #             :,
        #             pad_size_y : pad_size_y + fbank_fg.shape[-2],
        #             pad_size_x : pad_size_x + fbank_fg.shape[-1],
        #         ] = fbank_fg
        #         fbank_fg = temp
        #####[END] Original rescale and fit method.#####
        hr, wr = fbank_fg.shape[-2], fbank_fg.shape[-1]
        fbank_tmp = torch.zeros_like(fbank_fg)
        fbank_fg = F.interpolate(
            fbank_fg, (int(hr * resize_scale_y), int(wr * resize_scale_x))
            )
        pad_x = (wr - fbank_fg.shape[-1]) // 2
        pad_y = (hr - fbank_fg.shape[-2]) // 2
        px_tmp, py_tmp = max(pad_x, 0), max(pad_y, 0)
        px_tar, py_tar = max(-pad_x, 0), max(-pad_y, 0)

        fbank_tmp[:,:,py_tmp:fbank_fg.shape[-2]+py_tmp,px_tmp:fbank_fg.shape[-1]+px_tmp] = fbank_fg[
            :,:,py_tar:fbank_tmp.shape[-2]+py_tar,px_tar:fbank_tmp.shape[-1]+px_tar]
        fbank_fg = fbank_tmp

        latent_replace = self.editor.fbank2latent(fbank_fg)

        ddim_latents = self.editor.ddim_inv(
            latent=torch.cat([latent_base, latent_replace]),
            prompt=[prompt, prompt_replace],
        )
        latent_in = ddim_latents[-1][:1].squeeze(2)  # latent_base_noise

        # TODO: adapt it to different Gen models
        scale = 4 * SIZES[max(self.up_ft_index)] / self.up_scale
        edit_kwargs = process_paste(
            path_mask=mask_bg,
            h=f,
            w=t,
            dx=dx,
            dy=dy,
            scale=scale,
            input_scale=input_scale,
            up_scale=self.up_scale,
            up_ft_index=self.up_ft_index,
            w_edit=w_edit,
            w_content=w_content,
            precision=self.precision,
            resize_scale_x=resize_scale_x,
            resize_scale_y=resize_scale_y,
            )
        mask_tmp = (
            F.interpolate(
                edit_kwargs["mask_base_cur"].float(),
                (latent_in.shape[-2], latent_in.shape[-1]),
            )
            > 0
        ).float()
        # latent_replace_noise with rolling
        latent_tmp = torch.roll(
            ddim_latents[-1][1:].squeeze(2),
            (int(dy / (t / latent_in.shape[-2])), int(dx / (t / latent_in.shape[-2]))),
            (-2, -1),
        )
        latent_mix = slerp(bg_to_fg_ratio, latent_in, latent_tmp)
        latent_in = (latent_in * (1 - mask_tmp) + latent_mix * mask_tmp).to(
            dtype=latent_in.dtype
        )

        latent_rec = self.editor.pipe.edit(
            mode="mix",
            latent=latent_in,
            prompt=prompt,  # NOTE: emperically, make the rec the same as prompt base is the best
            guidance_scale=guidance_scale,
            energy_scale=energy_scale,
            latent_noise_ref=ddim_latents,
            SDE_strength=SDE_strength,
            edit_kwargs=edit_kwargs,
            disable_tangent_proj=disable_tangent_proj,
        )

        # Scale output latent
        if scale_denoised:
            _max = torch.max(torch.abs(latent_rec))
            latent_rec = latent_rec * 5 / _max

        spec_rec = self.editor.decode_latents(latent_rec)
        wav_rc = self.editor.mel_spectrogram_to_waveform(spec_rec)
        torch.cuda.empty_cache()

        return SoundEditorOutput(wav_rc, spec_rec)

    def run_remove(
        self,
        fbank_bg,
        mask_bg,
        fbank_fg,
        prompt,
        prompt_replace,
        w_edit,
        w_contrast,
        w_content,
        seed,
        guidance_scale,
        energy_scale,
        dx,
        dy,
        resize_scale_x,
        resize_scale_y,
        SDE_strength,
        save_kv=False,
        bg_to_fg_ratio=0.5,
        iterations=50,
        enable_penalty=True,
        disable_tangent_proj=False,
        scale_denoised=True,
    ):
        seed_everything(seed)
        energy_scale = energy_scale * 1e3

        # Prepare input spec and mask
        input_scale = 1
        fbank_bg = maybe_add_dimension(fbank_bg, 4).to(
            self.device, dtype=self.precision
        )  # shape = (B,C,T,F)
        f, t = fbank_bg.shape[-1], fbank_bg.shape[-2]
        fbank_fg = maybe_add_dimension(fbank_fg, 4).to(
            self.device, dtype=self.precision
        )

        if save_kv:
            self.editor.load_adapter()

        latent_base = self.editor.fbank2latent(fbank_bg)

        #####[START] Original rescale and fit method.#####
        # if resize_scale != 1:
        #     hr, wr = fbank_fg.shape[-2], fbank_fg.shape[-1]
        #     fbank_fg = F.interpolate(
        #         fbank_fg, (int(hr * resize_scale), int(wr * resize_scale))
        #     )
        #     pad_size_x = abs(fbank_fg.shape[-1] - wr) // 2
        #     pad_size_y = abs(fbank_fg.shape[-2] - hr) // 2
        #     if resize_scale > 1:
        #         fbank_fg = fbank_fg[
        #             :, :, pad_size_y : pad_size_y + hr, pad_size_x : pad_size_x + wr
        #         ]
        #     else:
        #         temp = torch.zeros(1, 1, hr, wr).to(self.device, dtype=self.precision)
        #         temp[
        #             :,
        #             :,
        #             pad_size_y : pad_size_y + fbank_fg.shape[-2],
        #             pad_size_x : pad_size_x + fbank_fg.shape[-1],
        #         ] = fbank_fg
        #         fbank_fg = temp
        #####[END] Original rescale and fit method.#####
        hr, wr = fbank_fg.shape[-2], fbank_fg.shape[-1]
        fbank_tmp = torch.zeros_like(fbank_fg)
        fbank_fg = F.interpolate(
            fbank_fg, (int(hr * resize_scale_y), int(wr * resize_scale_x))
            )
        pad_x = (wr - fbank_fg.shape[-1]) // 2
        pad_y = (hr - fbank_fg.shape[-2]) // 2
        px_tmp, py_tmp = max(pad_x, 0), max(pad_y, 0)
        px_tar, py_tar = max(-pad_x, 0), max(-pad_y, 0)

        fbank_tmp[:,:,py_tmp:fbank_fg.shape[-2]+py_tmp,px_tmp:fbank_fg.shape[-1]+px_tmp] = fbank_fg[
            :,:,py_tar:fbank_tmp.shape[-2]+py_tar,px_tar:fbank_tmp.shape[-1]+px_tar]
        fbank_fg = fbank_tmp

        latent_replace = self.editor.fbank2latent(fbank_fg)
        ddim_latents = self.editor.ddim_inv(
            latent=torch.cat([latent_base, latent_replace]),
            prompt=[prompt, prompt_replace],
        )
        latent_in = ddim_latents[-1][:1].squeeze(2)

        # TODO: adapt it to different Gen models
        scale = 4 * SIZES[max(self.up_ft_index)] / self.up_scale
        edit_kwargs = process_remove(
            path_mask=mask_bg,
            h=f,
            w=t,
            dx=dx,
            dy=dy,
            scale=scale,
            input_scale=input_scale,
            up_scale=self.up_scale,
            up_ft_index=self.up_ft_index,
            w_edit=w_edit,
            w_contrast=w_contrast,
            w_content=w_content,
            precision=self.precision,
            resize_scale_x=resize_scale_x,
            resize_scale_y=resize_scale_y,
        )
        mask_tmp = (
            F.interpolate(
                edit_kwargs["mask_base_cur"].float(),
                (latent_in.shape[-2], latent_in.shape[-1]),
            )
            > 0
        ).float()
        latent_tmp = torch.roll(
            ddim_latents[-1][1:].squeeze(2),
            (int(dy / (t / latent_in.shape[-2])), int(dx / (t / latent_in.shape[-2]))),
            (-2, -1),
        )
        # # F(B) <- F(M) - a * F(A)
        # latent_new = torch.randn_like(latent_tmp)
        # # latent_tmp = latent_tmp * latent_in.max()/latent_tmp.max() * 0.6 # 0.6 is the scale factor, a
        # m_ori, s_ori = latent_new.mean(dim=-2, keepdim=True), latent_new.std(dim=-2, keepdim=True)
        # # m_ref, s_ref = latent_tmp.mean(dim=-2, keepdim=True), latent_tmp.std(dim=-2, keepdim=True)
        # m_src, s_src = latent_in.mean(dim=-2, keepdim=True), latent_in.std(dim=-2, keepdim=True)
        # # s_new = torch.sqrt(s_src**2 - s_ref**2)
        # # latent_new = (latent_new - m_ori) / s_ori * s_new + (m_src - m_ref)
        # latent_new = (latent_new - m_ori) / s_ori * s_src + m_src

        # # Start from the latent of neighbor region
        # _m = mask_tmp.squeeze().sum(dim=1).nonzero().cpu()
        # stt_frame, end_frame = _m.min(), _m.max()
        # latent_neighbor = fill_with_neighbor(
        #     latent_in.squeeze(0), stt_frame, end_frame, neighbor_length=100
        # ) # 1s
        # __neighbor_energy_per_freq = (latent_neighbor*mask_tmp).mean(dim=0)
        # latent_neighbor[:,:,8:] *= 0.0001
        
        # Latent neighbor start from randomlized latent
        latent_neighbor = torch.randn_like(latent_in.squeeze(0)) * 0.9
        latent_neighbor = latent_neighbor + torch.randn_like(latent_neighbor) * 1e-3  # a little perturbation
        latent_neighbor, _ = optimize_neighborhood_points(
            latent_neighbor * mask_tmp,
            latent_tmp * mask_tmp,
            latent_in * mask_tmp,
            t=bg_to_fg_ratio,
            iterations=iterations,
            enable_penalty=enable_penalty,
            enable_tangent_proj=True,
        )  # TODO: try to turn off tangent
        latent_in = (latent_in * (1 - mask_tmp) + latent_neighbor * mask_tmp).to(
            dtype=latent_in.dtype
        )
        # latent_neighbor = torch.randn_like(latent_in) * 0.9
        # latent_in = (latent_in * (1 - mask_tmp) + latent_neighbor * mask_tmp).to(
        #     dtype=latent_in.dtype
        # )

        latent_rec = self.editor.pipe.edit(
            mode="remove",
            latent=latent_in,
            prompt=prompt,
            guidance_scale=guidance_scale,
            energy_scale=energy_scale,
            latent_noise_ref=ddim_latents,
            SDE_strength=SDE_strength,
            edit_kwargs=edit_kwargs,
            num_inference_steps=self.num_ddim_steps,
            start_time=self.num_ddim_steps,
            disable_tangent_proj=disable_tangent_proj,
        )

        # Scale output latent
        if scale_denoised:
            _max = torch.max(torch.abs(latent_rec))
            latent_rec = latent_rec * 5 / _max

        spec_rec = self.editor.decode_latents(latent_rec)
        wav_rc = self.editor.mel_spectrogram_to_waveform(spec_rec)
        torch.cuda.empty_cache()

        return SoundEditorOutput(wav_rc, spec_rec)

    def run_audio_generation(
        self,
        fbank_bg,
        mask_bg,
        fbank_fg,
        prompt,
        prompt_replace,
        w_edit,
        w_content,
        seed,
        guidance_scale,
        energy_scale,
        dx,
        dy,
        resize_scale_x,
        resize_scale_y,
        SDE_strength,
        save_kv=False,
        disable_tangent_proj=False,
        scale_denoised=True,
    ):
        seed_everything(seed)
        energy_scale = energy_scale * 1e3

        # Prepare input spec and mask
        input_scale = 1
        fbank_bg = maybe_add_dimension(fbank_bg, 4).to(
            self.device, dtype=self.precision
        )  # shape = (B,C,T,F)
        f, t = fbank_bg.shape[-1], fbank_bg.shape[-2]
        fbank_fg = maybe_add_dimension(fbank_fg, 4).to(
            self.device, dtype=self.precision
        )

        if save_kv:
            self.editor.load_adapter()

        latent_base = self.editor.fbank2latent(fbank_bg)

        #####[START] Original rescale and fit method.#####
        # if resize_scale != 1:
        #     hr, wr = fbank_fg.shape[-2], fbank_fg.shape[-1]
        #     fbank_fg = F.interpolate(
        #         fbank_fg, (int(hr * resize_scale), int(wr * resize_scale))
        #     )
        #     pad_size_x = abs(fbank_fg.shape[-1] - wr) // 2
        #     pad_size_y = abs(fbank_fg.shape[-2] - hr) // 2
        #     if resize_scale > 1:
        #         fbank_fg = fbank_fg[
        #             :, :, pad_size_y : pad_size_y + hr, pad_size_x : pad_size_x + wr
        #         ]
        #     else:
        #         temp = torch.zeros(1, 1, hr, wr).to(self.device, dtype=self.precision)
        #         temp[
        #             :,
        #             :,
        #             pad_size_y : pad_size_y + fbank_fg.shape[-2],
        #             pad_size_x : pad_size_x + fbank_fg.shape[-1],
        #         ] = fbank_fg
        #         fbank_fg = temp
        #####[END] Original rescale and fit method.#####
        hr, wr = fbank_fg.shape[-2], fbank_fg.shape[-1]
        fbank_tmp = torch.zeros_like(fbank_fg)
        fbank_fg = F.interpolate(
            fbank_fg, (int(hr * resize_scale_y), int(wr * resize_scale_x))
            )
        pad_x = (wr - fbank_fg.shape[-1]) // 2
        pad_y = (hr - fbank_fg.shape[-2]) // 2
        px_tmp, py_tmp = max(pad_x, 0), max(pad_y, 0)
        px_tar, py_tar = max(-pad_x, 0), max(-pad_y, 0)

        fbank_tmp[:,:,py_tmp:fbank_fg.shape[-2]+py_tmp,px_tmp:fbank_fg.shape[-1]+px_tmp] = fbank_fg[
            :,:,py_tar:fbank_tmp.shape[-2]+py_tar,px_tar:fbank_tmp.shape[-1]+px_tar]
        fbank_fg = fbank_tmp

        ddim_latents = self.editor.ddim_inv(
            latent=torch.cat([latent_base, latent_base]), prompt=[prompt, prompt]
        )
        latent_in = ddim_latents[-1][:1].squeeze(2)

        # TODO: adapt it to different Gen models
        scale = 4 * SIZES[max(self.up_ft_index)] / self.up_scale
        edit_kwargs = process_paste(
            path_mask=mask_bg,
            h=f,
            w=t,
            dx=dx,
            dy=dy,
            scale=scale,
            input_scale=input_scale,
            up_scale=self.up_scale,
            up_ft_index=self.up_ft_index,
            w_edit=w_edit,
            w_content=w_content,
            precision=self.precision,
            resize_scale_x=resize_scale_x,
            resize_scale_y=resize_scale_y,
        )

        latent_tmp = torch.randn_like(latent_in)
        mean, std = latent_in.mean(dim=-1, keepdim=True), latent_in.std(
            dim=-1, keepdim=True
        )
        m_ori, s_ori = latent_tmp.mean(dim=-1, keepdim=True), latent_tmp.std(
            dim=-1, keepdim=True
        )

        latent_tmp = (latent_tmp - m_ori) / s_ori * std + mean
        latent_in = latent_tmp

        latent_rec = self.editor.pipe.edit(
            mode="generate",
            latent=latent_in,
            prompt=prompt_replace,
            guidance_scale=guidance_scale,
            energy_scale=0,
            latent_noise_ref=ddim_latents,
            SDE_strength=SDE_strength,
            edit_kwargs=edit_kwargs,
            num_inference_steps=self.num_ddim_steps,
            start_time=self.num_ddim_steps,
            alg="D",
            disable_tangent_proj=disable_tangent_proj,
        )
        # Scale output latent
        if scale_denoised:
            _max = torch.max(torch.abs(latent_rec))
            latent_rec = latent_rec * 5 / _max

        spec_rec = self.editor.decode_latents(latent_rec)
        wav_rc = self.editor.mel_spectrogram_to_waveform(spec_rec)
        torch.cuda.empty_cache()

        return SoundEditorOutput(wav_rc, spec_rec)

    def run_style_transferring(
        self,
        fbank_bg,
        mask_bg,
        fbank_fg,
        prompt,
        prompt_replace,
        w_edit,
        w_content,
        seed,
        guidance_scale,
        energy_scale,
        dx,
        dy,
        resize_scale_x,
        resize_scale_y,
        SDE_strength,
        save_kv=True,
        disable_tangent_proj=False,
        scale_denoised=True,
    ):
        seed_everything(seed)
        energy_scale = energy_scale * 1e3

        # Prepare input spec and mask
        input_scale = 1
        fbank_bg = maybe_add_dimension(fbank_bg, 4).to(
            self.device, dtype=self.precision
        )  # shape = (B,C,T,F)
        f, t = fbank_bg.shape[-1], fbank_bg.shape[-2]

        if save_kv:
            self.editor.load_adapter()

        latent_base = self.editor.fbank2latent(fbank_bg)

        # if(torch.max(torch.abs(latent_base)) > 1e2):
        #     latent_base = torch.clip(latent_base, min=-10, max=10)

        ddim_latents = self.editor.ddim_inv(latent=latent_base, prompt=prompt,
                                            save_kv=True, mode="style_transfer",)
        latent_in = ddim_latents[-1].squeeze(2)

        scale = 4 * SIZES[max(self.up_ft_index)] / self.up_scale
        edit_kwargs = process_paste(
            path_mask=mask_bg,
            h=f,
            w=t,
            dx=dx,
            dy=dy,
            scale=scale,
            input_scale=input_scale,
            up_scale=self.up_scale,
            up_ft_index=self.up_ft_index,
            w_edit=w_edit,
            w_content=w_content,
            precision=self.precision,
            resize_scale_x=resize_scale_x,
            resize_scale_y=resize_scale_y,
        )

        # latent_tmp = torch.randn_like(latent_in)
        # mean, std = latent_in.mean(dim=-1, keepdim=True), latent_in.std(dim=-1, keepdim=True)
        # m_ori, s_ori = latent_tmp.mean(dim=-1, keepdim=True), latent_tmp.std(dim=-1, keepdim=True)

        # latent_tmp = (latent_tmp - m_ori) / s_ori * std + mean
        # latent_in = latent_tmp
        # import pdb; pdb.set_trace()
        latent_rec = self.editor.pipe.edit(
            mode="style_transfer",
            latent=latent_in,
            prompt=prompt_replace,
            guidance_scale=guidance_scale,
            energy_scale=energy_scale,
            latent_noise_ref=ddim_latents,
            SDE_strength=SDE_strength,
            edit_kwargs=edit_kwargs,
            num_inference_steps=self.num_ddim_steps,
            start_time=self.num_ddim_steps,
            alg="D",
            disable_tangent_proj=disable_tangent_proj,
        )

        # Scale output latent
        if scale_denoised:
            _max = torch.max(torch.abs(latent_rec))
            latent_rec = latent_rec * 5 / _max

        spec_rec = self.editor.decode_latents(latent_rec)
        wav_rc = self.editor.mel_spectrogram_to_waveform(spec_rec)
        torch.cuda.empty_cache()

        return SoundEditorOutput(wav_rc, spec_rec)

    def run_ddim_inversion(
        self,
        fbank_bg,
        mask_bg,
        fbank_fg,
        prompt,
        prompt_replace,
        w_edit,
        w_content,
        seed,
        guidance_scale,
        energy_scale,
        dx,
        dy,
        resize_scale_x,
        resize_scale_y,
        SDE_strength,
        save_kv=False,
        disable_tangent_proj=False,
        scale_denoised=True,
    ):
        seed_everything(seed)
        energy_scale = energy_scale * 1e3

        # Prepare input spec and mask
        input_scale = 1
        fbank_bg = maybe_add_dimension(fbank_bg, 4).to(
            self.device, dtype=self.precision
        )  # shape = (B,C,T,F)
        f, t = fbank_bg.shape[-1], fbank_bg.shape[-2]
        fbank_fg = maybe_add_dimension(fbank_fg, 4).to(
            self.device, dtype=self.precision
        )

        if save_kv:
            self.editor.load_adapter()

        latent_base = self.editor.fbank2latent(fbank_bg)

        if resize_scale != 1:
            hr, wr = fbank_fg.shape[-2], fbank_fg.shape[-1]
            fbank_fg = F.interpolate(
                fbank_fg, (int(hr * resize_scale), int(wr * resize_scale))
            )
            pad_size_x = abs(fbank_fg.shape[-1] - wr) // 2
            pad_size_y = abs(fbank_fg.shape[-2] - hr) // 2
            if resize_scale > 1:
                fbank_fg = fbank_fg[
                    :, :, pad_size_y : pad_size_y + hr, pad_size_x : pad_size_x + wr
                ]
            else:
                temp = torch.zeros(1, 1, hr, wr).to(self.device, dtype=self.precision)
                temp[
                    :,
                    :,
                    pad_size_y : pad_size_y + fbank_fg.shape[-2],
                    pad_size_x : pad_size_x + fbank_fg.shape[-1],
                ] = fbank_fg
                fbank_fg = temp

        # latent_replace = self.editor.fbank2latent(fbank_fg)
        ddim_latents = self.editor.ddim_inv(
            latent=torch.cat([latent_base, latent_base]), prompt=[prompt, prompt]
        )
        latent_in = ddim_latents[-1][:1].squeeze(2)

        # TODO: adapt it to different Gen models
        scale = 4 * SIZES[max(self.up_ft_index)] / self.up_scale
        edit_kwargs = process_paste(
            path_mask=mask_bg,
            h=f,
            w=t,
            dx=dx,
            dy=dy,
            scale=scale,
            input_scale=input_scale,
            up_scale=self.up_scale,
            up_ft_index=self.up_ft_index,
            w_edit=w_edit,
            w_content=w_content,
            precision=self.precision,
            resize_scale_x=resize_scale_x,
            resize_scale_y=resize_scale_y,
        )

        latent_rec = self.editor.pipe.edit(
            mode="generate",
            latent=latent_in,
            prompt=prompt_replace,
            guidance_scale=guidance_scale,
            energy_scale=0,
            latent_noise_ref=ddim_latents,
            SDE_strength=SDE_strength,
            edit_kwargs=edit_kwargs,
            num_inference_steps=self.num_ddim_steps,
            start_time=self.num_ddim_steps,
            alg="D",
            disable_tangent_proj=disable_tangent_proj,
        )

        # Scale output latent
        if scale_denoised:
            _max = torch.max(torch.abs(latent_rec))
            latent_rec = latent_rec * 5 / _max

        spec_rec = self.editor.decode_latents(latent_rec)
        wav_rc = self.editor.mel_spectrogram_to_waveform(spec_rec)
        torch.cuda.empty_cache()

        return SoundEditorOutput(wav_rc, spec_rec)


if __name__ == "__main__":
    mdl = AudioMorphix(
        "cvssp/audioldm-l-full", num_ddim_steps=50
    )  # "cvssp/audioldm-l-full" | "declare-lab/tango"
    print(mdl.__dict__)