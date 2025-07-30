import os
import matplotlib.pyplot as plt


from src.utils.factory import (
    plot_spectrogram,
)
from src.utils.audio_processing import extract_fbank, TacotronSTFT, maybe_add_dimension


def plot_single_spec(in_file, out_file):
    audio_or_path = in_file
    n_sample_per_sec = 100  # 1024frames / 10.24s

    fn_STFT = TacotronSTFT(
        filter_length=1024,
        hop_length=160,
        win_length=1024,
        n_mel_channels=64,
        sampling_rate=16000,
        mel_fmin=0,
        mel_fmax=8000,
    )

    fbank, log_stft, wav = extract_fbank(
        audio_or_path,
        fn_STFT=fn_STFT,
        target_length=1024,
        hop_size=160,
    )
    fbank = maybe_add_dimension(fbank, 3)  # (C,T,F)

    plot_spectrogram(fbank.permute(0, 2, 1)[:,:,:10*n_sample_per_sec], filename=out_file)
    plt.close('all')


file_paths = "/path/to/wav"
out_dir = "/path/to/spec"
os.makedirs(out_dir, exist_ok=True)

for root, dirs, files in os.walk(file_paths):
    for file in files:
        fpath = os.path.join(root, file)
        out_path = os.path.join(out_dir, file.replace(".wav", ".png"))
        plot_single_spec(in_file=fpath, out_file=out_path)