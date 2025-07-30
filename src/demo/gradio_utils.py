import spaces
import torch
import gradio as gr
import numpy as np

# Walk around container build-up
def get_installed_version(package="huggingface_hub"):
    import importlib
    try:
        module = importlib.import_module(package)
        return module.__version__
    except ImportError:
        return None
pkg_version = get_installed_version("huggingface_hub")
if pkg_version != "0.25.1":
    import sys
    import subprocess
    print(f"Installing huggingface_hub==0.25.1 (Current version: {pkg_version})")
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"huggingface_hub==0.25.1"])

from src.demo.utils import (
    create_model, get_spec_pil, get_mask_region, get_mask_regions, func_clear,
    update_reference_spec, get_spec_pil_with_original, get_spec_pils_for_moving,
    )
from src.audio_morphix import AudioMorphix
from src.utils.config import load_config
from src.demo.examples import mix_example, remove_example, moveandrescale_example


MODEL = AudioMorphix(pretrained_model_path="declare-lab/tango2-full", device="cpu")
CONFIG_FILE = "configs/runners/{}_runner.yaml"


@spaces.GPU(duration=90)
def run_add_task(
        cfgs,
        model, 
        background_fbank, 
        foreground_fbank, 
        background_text, 
        foreground_text, 
        mask_bg, 
        dt, df, 
        w_edit, 
        w_content, 
        guidance_scale, 
        energy_scale, 
        sde_strength, 
        resize_scale_t, 
        resize_scale_f, 
        seed,
    ):
    r"""Help function to feed the arguments to the model."""
    if torch.cuda.device_count() == 0:
        gr.Warning("Set this space to GPU config to make it fast.")
    model.to("cuda")

    processed_audio = model.run_mix(
        fbank_bg=background_fbank,
        mask_bg=mask_bg,
        fbank_fg=foreground_fbank,
        prompt=background_text,
        prompt_replace=foreground_text,
        dx=df, dy=dt,
        resize_scale_x=resize_scale_f,
        resize_scale_y=resize_scale_t,
        guidance_scale=guidance_scale,
        energy_scale=energy_scale,
        w_edit=w_edit,
        w_content=w_content,
        SDE_strength=sde_strength,
        seed=int(seed),
        save_kv=True,
        bg_to_fg_ratio=0.7,
        disable_tangent_proj=False,
        scale_denoised=False,
    ) 

    edited_wav = processed_audio.waveform
    if isinstance(edited_wav, np.ndarray):  # tango
        edited_wav = edited_wav[0]
    else:
        edited_wav = edited_wav.cpu().squeeze().numpy()

    return cfgs.audio_processor.sampling_rate, edited_wav


def create_add_demo():
    DESCRIPTION = """
    ## Audio Addition.
    Usage:
    - Upload a source audio and an appearance reference audio, and describe them separatly.
    - Mark the interested region on the source audio spectrum by brushing (on the FIRST layer).
    - Adjust the corresponding region on the reference audio spectrum.
    - Click the "Edit" button.
    """
    cfgs = load_config(CONFIG_FILE.format("mix"))

    # Claim variables
    config = gr.State(value=cfgs)
    model = gr.State(value=MODEL)
    foreground_fbank = gr.State(value=None)
    background_fbank = gr.State(value=None)
    foreground_spec_plot_ori = gr.State(value=None)
    src_mask = gr.State(value=None)
    ref_mask = gr.State(value=None)
    
    gr.Markdown(DESCRIPTION)
    with gr.Blocks() as demo:
        gr.Markdown(f"# Remove Task")
        gr.Markdown(f"## Select model type.")
        model_type = gr.Dropdown(
            value="declare-lab/tango2-full", 
            choices=["declare-lab/tango2-full","cvssp/audioldm-l-full","declare-lab/tango"],
            )
        model_type.select(fn=create_model, inputs=model_type, outputs=model)
        gr.Markdown(f"## Upload audio file(s) and describe audio content.")
        with gr.Row():
            with gr.Column():
                background_audio = gr.Audio(type="filepath", label="Background Audio")
                background_text = gr.Textbox(
                    label="Background audio description.", 
                    value=cfgs.task.background_audio_caption,
                    )
            with gr.Column():
                foreground_audio = gr.Audio(type="filepath", label="Foreground Audio")
                foreground_text = gr.Textbox(
                    label="Foreground audio description.", 
                    value=cfgs.task.foreground_audio_caption,
                    )
        gr.Markdown(f"## Mark the interseted regions.")
        gr.Markdown(f"Mark the region to edit by brushing on Layer 1.")
        with gr.Row():
            background_spec_plot = gr.ImageEditor(
                type="pil", label="Background spectrogram", 
                interactive=True,
                )
            foreground_spec_plot = gr.Image(
                type="pil", label="Foreground spectrogram", interactive=False)
        gr.Markdown("Tweak the following parameters to show and adjust the reference mask.")
        resize_scale_t = gr.Slider(0.5, 2.0, value=cfgs.task.resize_scale_t, label="Resize Scale (Time)")
        resize_scale_f = gr.Slider(0.5, 2.0, value=cfgs.task.resize_scale_f, label="Resize Scale (Frequency)")
        dt = gr.Slider(-1000, 1000, value=cfgs.task.dt, label="Time Shift (dt)")
        df = gr.Slider(-50, 50, value=cfgs.task.df, label="Frequency Shift (df)")
        with gr.Accordion("Advanced Options", open=False):
            w_edit = gr.Slider(0.0, 100.0, value=cfgs.task.w_edit, label="Edit Weight")
            w_content = gr.Slider(0.0, 100.0, value=cfgs.task.w_content, label="Content Weight")
            guidance_scale = gr.Slider(0.0, 10.0, value=cfgs.task.guidance_scale, label="Guidance Scale")
            energy_scale = gr.Slider(0.0, 5.0, value=cfgs.task.energy_scale, label="Energy Scale")
            sde_strength = gr.Slider(0.0, 1.0, value=cfgs.task.sde_strength, label="SDE Strength")
            seed = gr.Number(label="Random seed", value=cfgs.seed)
        gr.Markdown("## Output")
        output_audio = gr.Audio(label="Processed Audio")
        with gr.Row():
            run_button = gr.Button("Edit")
            clr_button = gr.Button("Clear")

        gr.Markdown("⬇️ Try our example(s) ⬇️")
        gr.Examples(
            examples=mix_example,
            inputs=[background_audio, background_text, foreground_audio, foreground_text, dt, df,
                    resize_scale_t, resize_scale_f, w_content, w_edit, guidance_scale, sde_strength, energy_scale]
        )

        # Handle event listener
        background_audio.change(
            get_spec_pil, 
            inputs=[model, background_audio, config], 
            outputs=[background_fbank, background_spec_plot],
            )
        foreground_audio.change(
            get_spec_pil_with_original, 
            inputs=[model, foreground_audio, config], 
            outputs=[foreground_fbank, foreground_spec_plot, foreground_spec_plot_ori],
            )
        background_spec_plot.change(
            fn=get_mask_region, 
            inputs=[background_spec_plot], 
            outputs=[src_mask],
            )
        src_mask.change(
            fn=update_reference_spec, 
            inputs=[foreground_spec_plot_ori, src_mask, dt, df, resize_scale_t, resize_scale_f], 
            outputs=[ref_mask, foreground_spec_plot],
        )
        dt.change(
            fn=update_reference_spec, 
            inputs=[foreground_spec_plot_ori, src_mask, dt, df, resize_scale_t, resize_scale_f], 
            outputs=[ref_mask, foreground_spec_plot],
            )
        df.change(
            fn=update_reference_spec, 
            inputs=[foreground_spec_plot_ori, src_mask, dt, df, resize_scale_t, resize_scale_f], 
            outputs=[ref_mask, foreground_spec_plot],
            )
        resize_scale_t.change(
            fn=update_reference_spec, 
            inputs=[foreground_spec_plot_ori, src_mask, dt, df, resize_scale_t, resize_scale_f], 
            outputs=[ref_mask, foreground_spec_plot],
            )
        resize_scale_f.change(
            fn=update_reference_spec, 
            inputs=[foreground_spec_plot_ori, src_mask, dt, df, resize_scale_t, resize_scale_f], 
            outputs=[ref_mask, foreground_spec_plot],
            )

        clr_button.click(
            fn=func_clear, 
            inputs=[background_audio, background_text, background_fbank, background_spec_plot, ref_mask, 
                    foreground_audio, foreground_text, foreground_fbank, foreground_spec_plot, foreground_spec_plot_ori, src_mask], 
            outputs=[background_audio, background_text, background_fbank, background_spec_plot, ref_mask, 
                    foreground_audio, foreground_text, foreground_fbank, foreground_spec_plot, foreground_spec_plot_ori, src_mask],
            )
        run_button.click(
            run_add_task, 
            inputs=[config, model, background_fbank, foreground_fbank, background_text, foreground_text, src_mask, dt, df, w_edit, w_content, guidance_scale, energy_scale, sde_strength, resize_scale_t, resize_scale_f, seed], 
            outputs=[output_audio],
            )
    return demo


@spaces.GPU(duration=90)
def run_remove_task(cfgs, model, background_fbank, foreground_fbank, background_text, foreground_text, mask_bg, dt, df, w_edit, w_content, w_contrast, guidance_scale, energy_scale, sde_strength, resize_scale_t, resize_scale_f, seed):
    r"""Help function to feed the arguments to the model."""  
    if torch.cuda.device_count() == 0:
        gr.Warning("Set this space to GPU config to make it fast.")
    model.to("cuda")
          
    processed_audio = model.run_remove(
        fbank_bg=background_fbank,
        mask_bg=mask_bg,
        fbank_fg=foreground_fbank,
        prompt=background_text,
        prompt_replace=foreground_text,
        dx=df, dy=dt,
        resize_scale_x=resize_scale_f,
        resize_scale_y=resize_scale_t,
        guidance_scale=guidance_scale,
        energy_scale=energy_scale,
        w_edit=w_edit,
        w_contrast=w_contrast,
        w_content=w_content,
        SDE_strength=sde_strength,
        seed=int(seed),
        save_kv=True,
        bg_to_fg_ratio=0.5,
        iterations=50,
        enable_penalty=True,
        disable_tangent_proj=False,
        scale_denoised=False,
    ) 

    edited_wav = processed_audio.waveform
    if isinstance(edited_wav, np.ndarray):  # tango
        edited_wav = edited_wav[0]
    else:
        edited_wav = edited_wav.cpu().squeeze().numpy()

    return cfgs.audio_processor.sampling_rate, edited_wav


def create_remove_demo():
    DESCRIPTION = """
    ## Audio Removal.
    Usage:
    - Upload a source audio and an appearance reference audio, and describe them separatly.
    - Mark the interested region on the source audio spectrum by brushing (on the FIRST layer).
    - Adjust the corresponding region on the reference audio spectrum.
    - Click the "Edit" button.
    """
    cfgs = load_config(CONFIG_FILE.format("remove"))

    # Claim variables
    config = gr.State(value=cfgs)
    model = gr.State(value=MODEL)
    foreground_fbank = gr.State(value=None)
    background_fbank = gr.State(value=None)
    foreground_spec_plot_ori = gr.State(value=None)
    src_mask = gr.State(value=None)
    ref_mask = gr.State(value=None)
    
    gr.Markdown(DESCRIPTION)
    with gr.Blocks() as demo:
        gr.Markdown(f"# Remove Task")
        gr.Markdown(f"## Select model type.")
        model_type = gr.Dropdown(
            value="declare-lab/tango2-full", 
            choices=["declare-lab/tango2-full","cvssp/audioldm-l-full","declare-lab/tango"],
            )
        # model_type.select(fn=create_model, inputs=model_type, outputs=model)
        gr.Markdown(f"## Upload audio file(s) and describe audio content.")
        with gr.Row():
            with gr.Column():
                background_audio = gr.Audio(type="filepath", label="Background Audio")
                background_text = gr.Textbox(
                    label="Background audio description.", 
                    value=cfgs.task.background_audio_caption,
                    )
            with gr.Column():
                foreground_audio = gr.Audio(type="filepath", label="Foreground Audio")
                foreground_text = gr.Textbox(
                    label="Foreground audio description.", 
                    value=cfgs.task.foreground_audio_caption,
                    )
        gr.Markdown(f"## Mark the interseted regions.")
        gr.Markdown(f"Mark the region to edit by brushing on Layer 1.")
        with gr.Row():
            background_spec_plot = gr.ImageEditor(
                type="pil", label="Background spectrogram", 
                interactive=True, canvas_size=(1024,64),
                )
            foreground_spec_plot = gr.Image(
                type="pil", label="Foreground spectrogram", interactive=False)
        gr.Markdown("Tweak the following parameters to show and adjust the reference mask.")
        resize_scale_t = gr.Slider(0.5, 2.0, value=cfgs.task.resize_scale_t, label="Resize Scale (Time)")
        resize_scale_f = gr.Slider(0.5, 2.0, value=cfgs.task.resize_scale_f, label="Resize Scale (Frequency)")
        dt = gr.Slider(-1000, 1000, value=cfgs.task.dt, label="Time Shift (dt)")
        df = gr.Slider(-50, 50, value=cfgs.task.df, label="Frequency Shift (df)")
        with gr.Accordion("Advanced Options", open=False):
            w_edit = gr.Slider(0.0, 100.0, value=cfgs.task.w_edit, label="Edit Weight")
            w_content = gr.Slider(0.0, 100.0, value=cfgs.task.w_content, label="Content Weight")
            w_contrast = gr.Slider(0.0, 5.0, value=cfgs.task.w_contrast, label="Contrast Weight")
            guidance_scale = gr.Slider(0.0, 10.0, value=cfgs.task.guidance_scale, label="Guidance Scale")
            energy_scale = gr.Slider(0.0, 5.0, value=cfgs.task.energy_scale, label="Energy Scale")
            sde_strength = gr.Slider(0.0, 1.0, value=cfgs.task.sde_strength, label="SDE Strength")
            seed = gr.Number(label="Random seed", value=cfgs.seed)
        gr.Markdown("## Output")
        output_audio = gr.Audio(label="Processed Audio")
        with gr.Row():
            run_button = gr.Button("Edit")
            clr_button = gr.Button("Clear")

        # Handle event listener
        background_audio.change(
            get_spec_pil, 
            inputs=[model, background_audio, config], 
            outputs=[background_fbank, background_spec_plot],
            )
        foreground_audio.change(
            get_spec_pil_with_original, 
            inputs=[model, foreground_audio, config], 
            outputs=[foreground_fbank, foreground_spec_plot, foreground_spec_plot_ori],
            )
        background_spec_plot.change(
            fn=get_mask_region, 
            inputs=[background_spec_plot], 
            outputs=[src_mask],
            )
        src_mask.change(
            fn=update_reference_spec, 
            inputs=[foreground_spec_plot_ori, src_mask, dt, df, resize_scale_t, resize_scale_f], 
            outputs=[ref_mask, foreground_spec_plot],
        )
        dt.change(
            fn=update_reference_spec, 
            inputs=[foreground_spec_plot_ori, src_mask, dt, df, resize_scale_t, resize_scale_f], 
            outputs=[ref_mask, foreground_spec_plot],
            )
        df.change(
            fn=update_reference_spec, 
            inputs=[foreground_spec_plot_ori, src_mask, dt, df, resize_scale_t, resize_scale_f], 
            outputs=[ref_mask, foreground_spec_plot],
            )
        resize_scale_t.change(
            fn=update_reference_spec, 
            inputs=[foreground_spec_plot_ori, src_mask, dt, df, resize_scale_t, resize_scale_f], 
            outputs=[ref_mask, foreground_spec_plot],
            )
        resize_scale_f.change(
            fn=update_reference_spec, 
            inputs=[foreground_spec_plot_ori, src_mask, dt, df, resize_scale_t, resize_scale_f], 
            outputs=[ref_mask, foreground_spec_plot],
            )

        gr.Markdown("⬇️ Try our example(s) ⬇️")
        gr.Examples(
            examples=remove_example,
            inputs=[background_audio, background_text, foreground_audio, foreground_text, dt, df,
                    resize_scale_t, resize_scale_f, w_content, w_edit, w_contrast, guidance_scale, sde_strength, energy_scale]
        )
        clr_button.click(
            fn=func_clear, 
            inputs=[background_audio, background_text, background_fbank, background_spec_plot, ref_mask, 
                    foreground_audio, foreground_text, foreground_fbank, foreground_spec_plot, foreground_spec_plot_ori, src_mask], 
            outputs=[background_audio, background_text, background_fbank, background_spec_plot, ref_mask, 
                    foreground_audio, foreground_text, foreground_fbank, foreground_spec_plot, foreground_spec_plot_ori, src_mask],
            )
        run_button.click(
            run_remove_task, 
            inputs=[config, model, background_fbank, foreground_fbank, background_text, foreground_text, src_mask, dt, df, w_edit, w_content, w_contrast, guidance_scale, energy_scale, sde_strength, resize_scale_t, resize_scale_f, seed], 
            outputs=[output_audio],
        )
    return demo


@spaces.GPU(duration=90)
def run_move_task(cfgs, model, background_fbank, background_text, mask_bg, mask_keep, dt, df, w_edit, w_content, w_contrast, w_inpaint, guidance_scale, energy_scale, sde_strength, resize_scale_t, resize_scale_f, seed):
    r"""Help function to feed the arguments to the model."""
    if torch.cuda.device_count() == 0:
        gr.Warning("Set this space to GPU config to make it fast.")
    model.to("cuda")

    processed_audio = model.run_move(
        fbank_org=background_fbank,
        mask=mask_bg,
        prompt=background_text,
        dx=df, dy=dt,
        mask_ref=None,
        mask_keep=mask_keep,
        resize_scale_x=resize_scale_f,
        resize_scale_y=resize_scale_t,
        guidance_scale=guidance_scale,
        energy_scale=energy_scale,
        w_edit=w_edit,
        w_contrast=w_contrast,
        w_content=w_content,
        w_inpaint=w_inpaint,
        SDE_strength=sde_strength,
        seed=int(seed),
        save_kv=True,
        disable_tangent_proj=False,
        scale_denoised=False,
    ) 

    edited_wav = processed_audio.waveform
    if isinstance(edited_wav, np.ndarray):  # tango
        edited_wav = edited_wav[0]
    else:
        edited_wav = edited_wav.cpu().squeeze().numpy()

    return cfgs.audio_processor.sampling_rate, edited_wav


def create_move_demo():
    DESCRIPTION = """
    ## Audio Moving and Rescaling.
    Usage:
    - Upload a source audio, and describe it separatly.
    - Mark the interested region on the source audio spectrum by brushing (on the FIRST layer).
    - Adjust the corresponding region on the reference audio spectrum.
    - Click the "Edit" button.
    """
    cfgs = load_config(CONFIG_FILE.format("move_and_resize"))

    # Claim variables
    config = gr.State(value=cfgs)
    model = gr.State(value=MODEL)
    foreground_fbank = gr.State(value=None)
    background_fbank = gr.State(value=None)
    foreground_spec_plot_ori = gr.State(value=None)
    src_mask = gr.State(value=None)
    ref_mask = gr.State(value=None)
    keep_mask = gr.State(value=None)
    
    gr.Markdown(DESCRIPTION)
    with gr.Blocks() as demo:
        gr.Markdown(f"# Move&Rescaling Task")
        gr.Markdown(f"## Select model type.")
        model_type = gr.Dropdown(
            value="declare-lab/tango2-full", 
            choices=["declare-lab/tango2-full","cvssp/audioldm-l-full","declare-lab/tango"],
            )
        # model_type.select(fn=create_model, inputs=model_type, outputs=model)
        gr.Markdown(f"## Upload audio file(s) and describe audio content.")
        with gr.Row():
            with gr.Column():
                background_audio = gr.Audio(type="filepath", label="Background Audio")
                background_text = gr.Textbox(
                    label="Background audio description.", 
                    value=cfgs.task.background_audio_caption,
                    )
        gr.Markdown(f"## Mark the interseted regions.")
        gr.Markdown(f"Mark the region to edit by brushing on Layer 1.")
        gr.Markdown(f"Mark the region to keep by brushing on Layer 2.")
        with gr.Row():
            background_spec_plot = gr.ImageEditor(
                type="pil", label="Background spectrogram", 
                interactive=True, canvas_size=(1024,64),
                )
            foreground_spec_plot = gr.Image(
                type="pil", label="Foreground spectrogram", interactive=False)
        gr.Markdown("Tweak the following parameters to show and adjust the reference mask.")
        resize_scale_t = gr.Slider(0.5, 2.0, value=cfgs.task.resize_scale_t, label="Resize Scale (Time)")
        resize_scale_f = gr.Slider(0.5, 2.0, value=cfgs.task.resize_scale_f, label="Resize Scale (Frequency)")
        dt = gr.Slider(-1000, 1000, value=cfgs.task.dt, label="Time Shift (dt)")
        df = gr.Slider(-50, 50, value=cfgs.task.df, label="Frequency Shift (df)")
        with gr.Accordion("Advanced Options", open=False):
            w_edit = gr.Slider(0.0, 100.0, value=cfgs.task.w_edit, label="Edit Weight")
            w_content = gr.Slider(0.0, 100.0, value=cfgs.task.w_content, label="Content Weight")
            w_contrast = gr.Slider(0.0, 5.0, value=cfgs.task.w_contrast, label="Contrast Weight")
            w_inpaint = gr.Slider(0.0, 5.0, value=cfgs.task.w_contrast, label="Contrast Weight")
            guidance_scale = gr.Slider(0.0, 10.0, value=cfgs.task.guidance_scale, label="Guidance Scale")
            energy_scale = gr.Slider(0.0, 5.0, value=cfgs.task.energy_scale, label="Energy Scale")
            sde_strength = gr.Slider(0.0, 1.0, value=cfgs.task.sde_strength, label="SDE Strength")
            seed = gr.Number(label="Random seed", value=cfgs.seed)
        gr.Markdown("## Output")
        output_audio = gr.Audio(label="Processed Audio")
        with gr.Row():
            run_button = gr.Button("Edit")
            clr_button = gr.Button("Clear")

        gr.Markdown("⬇️ Try our example(s) ⬇️")
        gr.Examples(
            examples=moveandrescale_example,
            inputs=[background_audio, background_text, dt, df,
                    resize_scale_t, resize_scale_f, w_content, w_edit, w_contrast, w_inpaint, 
                    guidance_scale, sde_strength, energy_scale]
        )

        # Event handler
        background_audio.change(
            get_spec_pils_for_moving, 
            inputs=[model, background_audio, config], 
            outputs=[background_fbank, background_spec_plot, foreground_fbank, 
                     foreground_spec_plot, foreground_spec_plot_ori],
            )
        background_spec_plot.change(
            fn=get_mask_regions, 
            inputs=[background_spec_plot], 
            outputs=[src_mask, keep_mask],
            )
        src_mask.change(
            fn=update_reference_spec, 
            inputs=[foreground_spec_plot_ori, src_mask, dt, df, resize_scale_t, resize_scale_f], 
            outputs=[ref_mask, foreground_spec_plot],
        )
        dt.change(
            fn=update_reference_spec, 
            inputs=[foreground_spec_plot_ori, src_mask, dt, df, resize_scale_t, resize_scale_f], 
            outputs=[ref_mask, foreground_spec_plot],
            )
        df.change(
            fn=update_reference_spec, 
            inputs=[foreground_spec_plot_ori, src_mask, dt, df, resize_scale_t, resize_scale_f], 
            outputs=[ref_mask, foreground_spec_plot],
            )
        resize_scale_t.change(
            fn=update_reference_spec, 
            inputs=[foreground_spec_plot_ori, src_mask, dt, df, resize_scale_t, resize_scale_f], 
            outputs=[ref_mask, foreground_spec_plot],
            )
        resize_scale_f.change(
            fn=update_reference_spec, 
            inputs=[foreground_spec_plot_ori, src_mask, dt, df, resize_scale_t, resize_scale_f], 
            outputs=[ref_mask, foreground_spec_plot],
            )

        clr_button.click(
            fn=func_clear, 
            inputs=[background_audio, background_text, background_fbank, background_spec_plot, ref_mask, 
                    foreground_fbank, foreground_spec_plot, foreground_spec_plot_ori, src_mask], 
            outputs=[background_audio, background_text, background_fbank, background_spec_plot, ref_mask, 
                    foreground_fbank, foreground_spec_plot, foreground_spec_plot_ori, src_mask],
            )
        run_button.click(
            run_move_task, 
            inputs=[config, model, background_fbank, background_text, src_mask, keep_mask, dt, df, w_edit, w_content, w_contrast, w_inpaint, guidance_scale, energy_scale, sde_strength, resize_scale_t, resize_scale_f, seed], 
            outputs=[output_audio],
        )
    return demo