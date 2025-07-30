import io
import copy
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import matplotlib.pyplot as plt
from PIL import Image

from src.audio_morphix import AudioMorphix
from src.utils.factory import plot_spectrogram, get_edit_mask
from src.utils.audio_processing import maybe_add_dimension


DESPLAY_RES = (1600, 900)
SPEC_RES = (1024, 64)
N_SAMPLE_PER_SEC = 100  # 1024frames / 10.24s


def func_clear(*args):
    result = []
    for arg in args:
        if isinstance(arg, list):
            result.append([])
        else:
            result.append(None)
    return tuple(result)


def create_model(model_type):
    model = AudioMorphix(pretrained_model_path=model_type, device="cpu")
    return model


def process_audio(model, audio, config):
    fbank, log_stft, wav = model.editor.get_fbank(
        audio, 
        config.audio_processor,
        return_intermediate=True,
        )
    fbank = maybe_add_dimension(fbank, 4)
    # Generate spectrogram plot
    spec_plot = plot_spectrogram(
        fbank.permute(0, 1, 3, 2)[:,:,:,:10*N_SAMPLE_PER_SEC], auto_amp=True)
    return fbank, spec_plot


def get_spec_pil(model, audio, config):
    try:
        fbank, spec_plot = process_audio(model, audio, config)
        buf = io.BytesIO()
        spec_plot.figure.savefig(buf, format='png')
        buf.seek(0)
        pil_spec = Image.open(buf)
        plt.close()
    except:
        print("Warning: the streaming is not ready. Please repeate uploading again.")
        fbank, pil_spec = None, None
    return fbank, pil_spec


def get_spec_pil_with_original(model, audio, config):
    fbank, pil_spec = get_spec_pil(model, audio, config)
    pil_spec_ori = copy.deepcopy(pil_spec)
    return fbank, pil_spec, pil_spec_ori


def get_spec_pils_for_moving(model, audio, config):
    src_fbank, src_pil_spec = get_spec_pil(model, audio, config)
    ref_fbank, ref_pil_spec = copy.deepcopy(src_fbank), copy.deepcopy(src_pil_spec)
    ref_pil_spec_ori = copy.deepcopy(ref_pil_spec)
    return src_fbank, src_pil_spec, ref_fbank, ref_pil_spec, ref_pil_spec_ori


def get_mask_region(img):
    layers = img['layers']
    if len(layers) > 0:
        print("Warning: Multiple layers exist while only the first layer is considered as the mask.")

    # Use the channel of opacity as mask
    mask = pil_to_tensor(layers[0])[-1,:,:]  # RGBA
    mask = mask.permute(1, 0)  # (F, T) -> (T, F)
    # Flip the freq axis to ensure the orignal point on the top left
    mask = mask.flip(1)
    mask = (mask > 0).float()

    # Rescale mask to spectrum size
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), SPEC_RES).squeeze()
    return mask


def get_mask_regions(img):
    def _prepare_mask(m):
        m = m.permute(1, 0)
        # Flip the freq axis to ensure the orignal point on the top left
        m = m.flip(1)
        m = (m > 0).float()
        m = F.interpolate(m.unsqueeze(0).unsqueeze(0), SPEC_RES).squeeze()
        return m

    layers = img['layers']
    if len(layers) > 0:
        print("Warning: Multiple layers exist while the first layer is considered as the mask to edit and the second is the mask to keep.")

    if len(layers) > 1:
        mask_src = pil_to_tensor(layers[0])[-1,:,:]  # RGBA
        mask_keep = pil_to_tensor(layers[1])[-1,:,:]
        mask_src, mask_keep = _prepare_mask(mask_src), _prepare_mask(mask_keep)
    elif len(layers) == 1:
        mask_src = pil_to_tensor(layers[0])[-1,:,:]
        mask_src = _prepare_mask(mask_src)
        mask_keep = None
    else:
        mask_src, mask_keep = None, None

    return mask_src, mask_keep



def update_reference_spec(ref_spec_pil_ori, mask_src, dt, df, resize_scale_t, resize_scale_f):
    if mask_src is not None:
        mask_ref = get_edit_mask(
            mask_src, dx=df, dy=dt, 
            resize_scale_x=resize_scale_f, 
            resize_scale_y=resize_scale_t,
            )
        mask_ref = mask_ref.float()  # match the PIL format, channel last
        mask_ref_pil = F.interpolate(mask_ref.unsqueeze(0).unsqueeze(0), DESPLAY_RES).squeeze()

        # Match the shape to the PIL format (H, W, C)
        if mask_ref_pil.ndim > 2:
            mask_ref_pil = mask_ref_pil.squeeze()
        mask_ref_pil = mask_ref_pil.permute(1, 0)
        # De-flip freq exis to match pil imshow style
        mask_ref_pil = mask_ref_pil.flip(0)
        mask_ref_pil = mask_ref_pil * 0.5  # for transparency

        # Convert to PIL
        mask_ref_pil = to_pil_image(mask_ref_pil).convert("L")
        # mask_ref_pil = mask_ref_pil.resize(ref_spec_pil_ori.size)

        overlay = Image.new("RGBA", mask_ref_pil.size, (128, 255, 255, 50))  # create overlay
        ref_spec_pil = Image.composite(overlay, ref_spec_pil_ori, mask_ref_pil)
    else:
        ref_spec_pil = ref_spec_pil_ori
        mask_ref = None
        
    return mask_ref, ref_spec_pil