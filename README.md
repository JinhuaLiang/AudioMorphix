# 🔉 AudioMorphix: Training-free Audio Editing with Diffusion Models

<div align="center">
[![Demo](https://img.shields.io/badge/🌐%20Demo-AudioMorphix-green)](https://jinhualiang.github.io/AudioMorphix-Demo/)
[![arXiv](https://img.shields.io/badge/arXiv-2505.16076-B31B1B.svg)](https://arxiv.org/abs/2505.16076)
<!-- [![Paper](https://img.shields.io/badge/Paper-OpenReview-blue)](https://openreview.net/forum?id=a8dQutiF9E) -->
[![HF Space](https://img.shields.io/badge/🤗%20HF%20Space-AudioMorphix-orange)](https://huggingface.co/spaces/JinhuaL1ANG/AudioMorphix)

*Training-free audio editing with diffusion probabilistic models*

</div>

## 📖 Description

AudioMorphix is a **training-free audio editor** that enables precise audio editing using diffusion models. Unlike traditional text-based audio editing methods, AudioMorphix uses **audio-referenced editing**, allowing you to manipulate target regions on spectrograms by referencing other audio recordings.

### Key Features:

- 🚀 **Zero-shot editing**: No additional training required for task-specific data
- 🎯 **Precise control**: Manipulate specific regions in time and frequency domains
- 🔄 **Interactive editing**: Real-time visual feedback with spectrogram-based region selection
- 🎨 **Multiple tasks**: Supports addition, removal, time shifting, stretching, and pitch shifting

## 🌟 Capabilities

AudioMorphix demonstrates promising performance on various audio editing tasks:

- **🎶 Mix Audio**: Add foreground sounds to background audio
- **🔇 Remove Audio**: Remove specific sounds from audio recordings
- **⏰ Move & Resize**: Time shifting and frequency scaling of audio elements
- **🎵 Style Transfer**: Apply audio characteristics from one source to another

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/JinhuaLiang/AudioMorphix.git
cd AudioMorphix
pip install -r requirements.txt
```

**Optional: Install with conda environment**

```bash
conda create -n audiomorphix python=3.8
conda activate audiomorphix
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0.1 + CUDA 11.8

## 🎛️ Usage

### 🌐 Web Application

Launch the interactive Gradio interface:

```bash
python3 app.py
```

The web app will be available at `http://localhost:7200` and provides an intuitive interface with:

- **🎶 Mix Audio Tab**: Combine foreground and background audio with visual spectrogram editing
- **🔇 Remove Audio Tab**: Remove specific audio elements by selecting regions
- **⏰ Move & Resize Audio Tab**: Manipulate audio positioning and scaling with drag-and-drop controls
- **📊 Real-time Preview**: Instant audio and spectrogram feedback
- **🎛️ Parameter Controls**: Fine-tune guidance scale, energy scale, and other parameters

### 💻 Command Line Interface

Run audio editing tasks from the command line:

```bash
python3 inference.py --config configs/runners/[TASK]_runner.yaml
```

**Available task configurations:**

- `mix_runner.yaml` - Audio mixing and layering
- `remove_runner.yaml` - Audio element removal
- `move_and_resize_runner.yaml` - Temporal and spectral transformations
- `style_transfer_runner.yaml` - Audio style transfer

**Some examples**

```bash
# Mix Audio Example
python3 inference.py \
    --config configs/runners/mix_runner.yaml \
    --task.background_audio_filepath "examples/Thunder and a gentle rain.wav" \
    --task.foreground_audio_filepath "examples/dog_barking.wav" \
    --task.background_audio_caption "Thunder and gentle rain" \
    --task.foreground_audio_caption "Dog barking"

# Remove Audio Example
python3 inference.py \
    --config configs/runners/remove_runner.yaml \
    --task.background_audio_filepath "examples/tick_noise_with_laughter.wav" \
    --task.foreground_audio_filepath "examples/tick_noise-noised.wav" \
    --task.background_audio_caption "tick noise with laughter" \
    --task.foreground_audio_caption "tick noise"
```

### Python API

```python
from src.audio_morphix import AudioMorphix

# Initialize the model
model = AudioMorphix(pretrained_model_path="declare-lab/tango2-full")

# Load and process audio
fbank_bg, _, wav_bg = model.editor.get_fbank(
    "examples/Thunder and a gentle rain.wav", config.audio_processor
)
fbank_fg, _, wav_fg = model.editor.get_fbank(
    "examples/dog_barking.wav", config.audio_processor
)

# Define editing region
mask_bg = torch.zeros_like(fbank_bg)
mask_bg[t_start:t_end, f_low:f_high] = 1

# Perform mixing operation
result = model.run_mix(
    fbank_bg=fbank_bg,
    mask_bg=mask_bg,
    fbank_fg=fbank_fg,
    prompt="background audio description",
    prompt_replace="foreground audio description",
    guidance_scale=7.5,
    energy_scale=1.0
)

# Get edited audio
edited_audio = result.waveform
```

## 📁 Project Structure

```
AudioMorphix/
├── app.py                    # Gradio web interface
├── inference.py              # Command-line inference script
├── src/
│   ├── audio_morphix.py      # Main AudioMorphix class
│   ├── model/                # Model components
│   ├── demo/                 # Demo utilities
│   └── utils/                # Utility functions
├── configs/                  # Configuration files
│   ├── tasks/               # Task-specific configs
│   └── runners/             # Runner configurations
├── examples/                 # Example audio files
└── scripts/                 # Shell scripts
```

## 🔧 Advanced Configuration

AudioMorphix uses YAML configuration files for detailed control over editing parameters:

### Core Parameters

- **`guidance_scale`** (default: 1.2): Controls the strength of text conditioning
- **`energy_scale`** (default: 0.5): Adjusts energy preservation during editing
- **`sde_strength`** (default: 0.4): Controls the amount of noise injection for stochastic editing
- **`num_ddim_steps`** (default: 50): Number of denoising steps

### Edit Control Weights

- **`w_content`** (default: 10.0): Weight for content preservation
- **`w_edit`** (default: 15.0): Weight for editing guidance
- **`w_contrast`** (default: 0.5): Weight for contrast enhancement (removal tasks)

### Region Selection

- **`t_on`/`t_off`**: Time boundaries for editing region
- **`f_low`/`f_up`**: Frequency boundaries for editing region
- **`resize_scale_t`/`resize_scale_f`**: Temporal and spectral scaling factors
- **`df`/`dt`**: Shift distance in frequency and time dimensions

**Example configuration:**

```yaml
task:
  guidance_scale: 1.2
  energy_scale: 0.5
  w_content: 10
  w_edit: 15
  sde_strength: 0.4
  num_ddim_steps: 50
```

Refer to the `configs/` directory for detailed parameter explanations and task-specific examples.

## 📊 Performance & Benchmarks

AudioMorphix achieves high-fidelity audio editing across various tasks:

### Quality Metrics

- **Temporal Precision**: high accuracy in time-based edits
- **Spectral Fidelity**: Preserves content of original frequency characteristics
- **Seamless Integration**: Smooth transitions with minimal artifacts
- **Content Preservation**: Maintains non-edited regions with high similarity

### Supported Audio Types

- **Speech**: monologues
- **Music**: Instruments, mixed tracks
- **Sound Effects**: Nature sounds, synthetic sounds

### Performance Characteristics

- **Processing Speed**: ~90 seconds per 10-second audio clip (GPU)
- **Memory Usage**: ~18GB VRAM for typical edits

## 📄 Citation

If you use AudioMorphix in your research, please cite:

```bibtex
@article{liang2024audiomorphix,
  title={AudioMorphix: Training-free audio editing with diffusion probabilistic models},
  author={Liang, Jinhua and others},
  journal={arXiv preprint arXiv:2505.16076},
  year={2024}
}
```

## 🙏 Acknowledgments

This project builds upon several excellent open-source projects:

- [**Tango/Tango2**](https://github.com/declare-lab/tango) - Base diffusion model architecture
- [**AudioLDM**](https://github.com/haoheliu/AudioLDM) - Audio latent diffusion framework
- [**DragonDiffusion**](https://github.com/haoheliu/AudioLDM) - Diffusion-based editing frameworks
- [**Diffusers**](https://github.com/huggingface/diffusers) - Diffusion model utilities and pipelines
- [**Hugging Face**](https://huggingface.co/) - Model hosting, distribution, and inference

Special thanks to Hugging Face for the community grant.

## 📝 License

This project is licensed under the NonCommercial License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

For questions and feedback:

- 📧 **Email**: liangjh0903@gmail.com
- 💬 **Discussions**: [GitHub Discussions](https://github.com/JinhuaLiang/AudioMorphix/discussions)
