import os
import torch
import torchaudio
from torch import Tensor
from typing import Any, Callable, List
from random import randint, uniform, betavariate


class NaiveAudioProcessor:
    __doc__ = r"""A naive processor for audio processor."""

    def __call__(self, filename, filename2=None):
        return self.extract_features(filename, filename2)

    def extract_features(self, filename, filename2=None):
        r"""Dummy func to extract features."""
        return {}

    @staticmethod
    def torchaudio_to_byte(
        audio: torch.Tensor,
        sampling_rate: int,
        cache_path="./.tmp.flac",
    ):
        torchaudio.save(
            filepath=cache_path,
            src=audio,
            sample_rate=sampling_rate,
        )

        with open(cache_path, "rb") as f:
            audio_stream = f.read()

        os.remove(cache_path)

        return audio_stream


class WaveformAudioProcessor(NaiveAudioProcessor):
    __doc__ = r"""A processor to load wavform from audio files."""

    def __init__(
        self,
        sampling_rate: int = 16000,
        duration: float = 10.24,
        normalize: bool = True,
        trim_wav: bool = True,
        transforms: List[Callable] = [],
    ):
        self.sampling_rate = sampling_rate
        self.audio_duration = duration
        self.normalize = normalize
        self.trim_wav = trim_wav
        # Data augmentation
        self.transforms = transforms

    def extract_features(self, filename, filename2=None):
        r"""Return waveform."""
        wav = self.load_wav(
            filename,
            sampling_rate=self.sampling_rate,
            normalize=self.normalize,
            trim_wav=self.trim_wav,
        )

        # Mix two wavs if `filename2` is given
        if filename2 is not None:
            wav2 = self.load_wav(
                filename2,
                sampling_rate=self.sampling_rate,
                normalize=self.normalize,
                trim_wav=self.trim_wav,
            )
            mixture, mix_lambda = self.mix_wavs(wav, wav2)
        else:
            mixture = wav

        # Data augmentatioin if applicable
        if len(self.transforms) > 0:
            for transform in self.transforms:
                mixture = transform(mixture)

        return {"waveform": mixture}

    def load_wav(
        self,
        wav_file: str,
        sampling_rate: int = 16000,
        normalize: bool = True,
        trim_wav: bool = False,
    ) -> list:
        r"""Return (torch.Tensor, float), Tensor shape = (c, n_temporal_step)."""
        audio, sr = torchaudio.load(wav_file)

        # Resample the audio if `resample` = True
        if sr != sampling_rate:
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=sr,
                new_freq=self.sampling_rate,
            )

        # Detect the activate clip from audio if `trim_wav` = True
        if trim_wav:
            audio = self.maybe_trim_wav(audio)

        # Uniform the length of output wavs
        target_length = int(self.audio_duration * sr)
        if len(audio) < target_length:
            audio = self.pad_wav(audio, target_length, pad_last=True)
        elif len(audio) > target_length:
            audio = self.segment_wav(audio, target_length, truncation="right")

        # Z-nomalize the output wavs if `normalize` = True
        if normalize:
            try:
                audio = self.normalize_wav(audio)
            except RuntimeError as e:
                print(f"{e}: {wav_file} is empty.")

        return audio

    @staticmethod
    def normalize_wav(waveform: Tensor, eps=torch.tensor(1e-8)):
        r"""Return wavform with mean=0, std=0.5."""
        waveform = waveform - waveform.mean()
        waveform = waveform / torch.max(waveform.abs() + eps)

        return waveform * 0.5  # manually limit the maximum amplitude into 0.5

    @staticmethod
    def mix_wavs(waveform1, waveform2, alpha=10, beta=10):
        mix_lambda = betavariate(alpha, beta)
        mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2

        return __class__.normalize_wav(mix_waveform), mix_lambda

    @staticmethod
    def split_wavs(waveform, target_length, padding_mode="zeros"):
        r"""Split wav into several pieces with the length `target_length`.
        Args: `waveform` is a 2d channel-first tensor.
        """
        segmented_wavs = []
        n_channels, wav_length = waveform.size()
        for stt_idx in range(0, wav_length, target_length):
            end_idx = stt_idx + target_length
            if end_idx > wav_length:
                # NOTE: Drop the last seg if it is too short
                if (wav_length - stt_idx) < 0.1 * target_length:
                    break
                # Pad the last seg with the content in the previous one
                if padding_mode == "replicate":
                    segmented_wavs.append(waveform[:, -target_length:])
                else:
                    assert padding_mode == "zeros"
                    _tmp_wav = waveform[:, stt_idx:]
                    _padded_wav = torch.zeros(n_channels, wav_length)
                    _padded_wav[:, : _tmp_wav.size(dim=-1)] += _tmp_wav
                    segmented_wavs.append(_padded_wav)
            else:
                segmented_wavs.append(waveform[:, stt_idx:end_idx])

        return segmented_wavs

    @staticmethod
    def segment_wav(
        waveform,
        target_length,
        truncation="right",
    ):
        r"""Return semented wav of `target_length` and the start time of the segmentation."""
        assert truncation in ["left", "right", "random"]

        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        # Too short
        if (waveform_length - target_length) <= 0:
            return waveform, 0

        # Try at most 10 times to find a valid start index
        for i in range(10):
            if truncation == "left":
                start_index = waveform_length - target_length
            elif truncation == "right":
                start_index = 0
            else:
                start_index = randint(0, waveform_length - target_length)

            if torch.max(
                torch.abs(waveform[:, start_index : start_index + target_length]) > 1e-4
            ):
                break

        return waveform[:, start_index : start_index + target_length], start_index

    @staticmethod
    def pad_wav(waveform, target_length, pad_last=True):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, {waveform_length}"

        if waveform_length == target_length:
            return waveform

        # Pad
        output_wav = torch.zeros((1, target_length), dtype=torch.float32)

        if not pad_last:
            rand_start = randint(0, target_length - waveform_length)
        else:
            rand_start = 0

        output_wav[:, rand_start : rand_start + waveform_length] = waveform
        return output_wav

    @staticmethod
    def maybe_trim_wav(waveform):
        r"""Trim the wav by remove the silence part."""
        if waveform.abs().max() < 0.0001:
            return waveform

        def detect_leading_silence(waveform, threshold=1e-4):
            chunk_size = 1000
            waveform_length = waveform.shape[0]

            start = 0
            while start + chunk_size < waveform_length:
                if waveform[start : start + chunk_size].abs().max() < threshold:
                    start += chunk_size
                else:
                    break

            return start

        def detect_ending_silence(waveform, threshold=1e-4):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = waveform_length

            while start - chunk_size > 0:
                if waveform[start - chunk_size : start].abs().max() < threshold:
                    start -= chunk_size
                else:
                    break

            if start == waveform_length:
                return start
            else:
                return start + chunk_size

        start = detect_leading_silence(waveform)
        end = detect_ending_silence(waveform)

        return waveform[start:end]


class FbankAudioProcessor(WaveformAudioProcessor):
    def __init__(
        self,
        # Fbank setting
        n_frames: int = 1024,
        n_mels: int = 128,
        # Waveform setting
        sampling_rate: int = 16000,
        duration: float = 10.24,
        normalize: bool = True,
        trim_wav: bool = True,
        # Data augmentation
        transforms: List[Callable] = [],
    ):
        super().__init__(sampling_rate, duration, normalize, trim_wav)
        self.n_frames = n_frames
        self.n_mels = n_mels
        # Data augmentation
        self.transforms = transforms

    def extract_features(self, filename, filename2=None):
        wav = self.load_wav(
            filename,
            sampling_rate=self.sampling_rate,
            normalize=self.normalize,
            trim_wav=self.trim_wav,
        )

        # Mix two wavs if `filename2` is given
        if filename2 is not None:
            wav2 = self.load_wav(
                filename2,
                sampling_rate=self.sampling_rate,
                normalize=self.normalize,
                trim_wav=self.trim_wav,
            )
            mixture, mix_lambda = self.mix_wavs(wav, wav2)
        else:
            mixture = wav

        # Get fbank from the `mixture`
        # shape of `fbank` = (`n_frames`, `n_mels`)
        fbank = self.wav2fbank(
            mixture,
            self.n_frames,
            self.n_mels,
            self.sampling_rate,
        )

        # Transform fbank for data augemtnation if applicable
        if len(self.transforms) > 0:
            for transform in self.transforms:
                fbank = transform(fbank)

        return {"waveform": mixture, "fbank": fbank}

    def wav2fbank(
        self,
        wav,
        n_frames=1024,
        n_mels=128,
        sampling_rate=16000,
        norm_mean=-4.2677393,
        norm_std=4.5689974,
    ):
        try:
            fbank = torchaudio.compliance.kaldi.fbank(
                wav,
                htk_compat=True,
                sample_frequency=sampling_rate,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=n_mels,
                dither=0.0,
                frame_shift=10,
            )
        except AssertionError as e:
            fbank = torch.zeros([n_frames, n_mels]) + 0.01
            print(f"A empty fbank loaded as {e}.")

        # Cut and pad to the length of `n_frames`
        return self.pad_or_clip_fbank(fbank, n_frames)

    @staticmethod
    def pad_fbank(fbank, padding_length):
        m = torch.nn.ZeroPad2d((0, 0, 0, padding_length))
        return m(fbank)

    @staticmethod
    def clip_fbank(fbank, target_length):
        return fbank[0:target_length, :]

    @staticmethod
    def pad_or_clip_fbank(fbank, target_length):
        p = target_length - fbank.shape[0]  # target_length - curr_n_frames
        if p > 0:
            return __class__.pad_fbank(fbank, p)
        else:
            return __class__.clip_fbank(fbank, target_length)


class AddGaussianNoise:
    def __init__(self, noise_magnitude=uniform(0, 1) * 0.1):
        self.noise_magnitude = noise_magnitude

    def __call__(self, fbank):
        d0, d1 = fbank.size()
        return fbank + torch.rand(d0, d1) * self.noise_magnitude


class TimeRolling:
    def __init__(self, rolling_step=None):
        self.rs = rolling_step

    def __call__(self, fbank):
        return torch.roll(fbank, randint(-self.rs, self.rs - 1), 0)


class FbankTimeMasking:
    __doc__ = r"""Masking the time dimension of fbank for data augmentation
    with the length ranged of (0, `timem`)."""

    def __init__(self, timem: int = 0):
        from torchaudio.transforms import TimeMasking

        self.mask_fn = TimeMasking(timem)

    def __call__(self, fbank) -> Tensor:
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)

        fbank = self.mask_fn(fbank)

        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)
        return fbank


class FbankFrequencyMasking:
    __doc__ = r"""Masking the frequency dimension of fbank for data augmentation
    with the length ranged of (0, `freqm`)."""

    def __init__(self, freqm: int = 0):
        from torchaudio.transforms import FrequencyMasking

        self.mask_fn = FrequencyMasking(freqm)

    def __call__(self, fbank) -> Tensor:
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)

        fbank = self.mask_fn(fbank)

        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)
        return fbank


class SpecAugment:
    __doc__ = r"""Masking the time & frequency dimension of fbank for data augmentation
    with the length ranged of (0, `timem`), (0, `freqm`), respectively."""

    def __init__(self, timem: int = 0, freqm: int = 0) -> None:
        from torchaudio.transforms import TimeMasking, FrequencyMasking

        self.time_mask_fn = TimeMasking(timem)
        self.freq_mask_fn = FrequencyMasking(freqm)

    def __call__(self, fbank) -> Tensor:
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)

        fbank = self.time_mask_fn(fbank)
        fbank = self.freq_mask_fn(fbank)

        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)
        return fbank


if __name__ == "__main__":
    import debugger

    wav_file = (
        "/mnt/bn/lqhaoheliu/datasets/audioset/zip_audios/eval_segments/Y-53zl3bPmpM.wav"
    )
    wav_file2 = (
        "/mnt/bn/lqhaoheliu/datasets/audioset/zip_audios/eval_segments/Y-6Aq2fJwlgU.wav"
    )

    audio_processor = FbankAudioProcessor(transforms=[SpecAugment(1024, 128)])
    # print(audio_processor(wav_file, wav_file2)[0].shape)
    # print(audio_processor(wav_file, wav_file2)[1].shape)
    print(audio_processor(wav_file, wav_file2)[1])
