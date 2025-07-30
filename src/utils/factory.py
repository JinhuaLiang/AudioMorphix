import librosa
import torch
import json
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch import lerp
from torch.nn import ReflectionPad1d
import torch.nn.functional as F


def load_json(fname):
    with open(fname, "r") as f:
        data = json.load(f)
        return data


# def plot_spectrogram(fbank, filename=None, title=None, ylabel="freq_bin", ax=None):
#     r"""
#     Params: `fbank`: (`n_mel_bins`, `n_frames`)
#     """
#     if fbank.ndim > 2:
#         fbank = fbank.detach().cpu().squeeze()
#     else:
#         fbank = fbank.detach().cpu()
#     if ax is None:
#         _, ax = plt.subplots(1, 1)
#     if title is not None:
#         ax.set_title(title)
#     ax.set_ylabel(ylabel)
#     ax.imshow(fbank, origin="lower", aspect="auto", interpolation="nearest")
#     if filename is not None:
#         ax.figure.savefig(filename)
#     return ax
def plot_spectrogram(fbank, filename=None, title=None, ylabel=None, auto_amp=False, figsize=(16, 9)):
    r"""
    Params: `fbank`: (`n_mel_bins`, `n_frames`)
    """
    if fbank.ndim > 2:
        fbank = fbank.detach().cpu().squeeze()
    else:
        fbank = fbank.detach().cpu()
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    fbank = fbank.numpy()

    if auto_amp:
        img=librosa.display.specshow(fbank, ax=ax)
    else:
        img=librosa.display.specshow(fbank, ax=ax, vmin=-10, vmax=0)  # x_axis='time', y_axis='mel', 

    if title is not None:
        ax.set_title(title)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust subplots to fill the figure

    if filename is not None:
        ax.figure.savefig(filename)
    return ax


def get_current_time(out_format="%Y-%m-%d %H:%M:%S"):
    current_time = datetime.now()
    formatted_time = current_time.strftime(out_format)
    return formatted_time


def get_box_boundry(mask: torch.Tensor):
    r"""Get the box boundy of masked region."""
    ws, hs = torch.nonzero(mask, as_tuple=True)

    w_l, w_r = torch.min(ws), torch.max(ws)
    h_b, h_t = torch.min(hs), torch.max(hs)

    return (w_l, w_r), (h_b, h_t)


def get_neibor_with_mask(matrix, mask, reverse=False):
    assert matrix.shape == mask.shape

    # Pad the unmasked region using reflection if applicable
    if reverse:
        mask = ~mask.bool()

    (w_l, w_r), (h_b, h_t) = get_box_boundry(mask)

    mask_w_cntr = (w_r + w_l) // 2

    pad_l_fn = ReflectionPad1d((0, mask_w_cntr - w_l))
    matrix_l_cur = pad_l_fn(matrix[: w_l + 1, :].permute(1, 0)).permute(1, 0)
    pad_r_fn = ReflectionPad1d((w_r - mask_w_cntr - 1))
    import ipdb

    ipdb.set_trace()
    matrix_r_cur = pad_r_fn(matrix[w_r + 1 :, :].permute(1, 0)).permute(1, 0)
    # import ipdb; ipdb.set_trace()
    matrix_cur = torch.cat([matrix_l_cur, matrix_r_cur], dim=0)

    return matrix[mask] + matrix_cur[~mask]

    # def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    #     '''
    #     Spherical linear interpolation
    #     Args:
    #         t (float/np.ndarray): Float value between 0.0 and 1.0
    #         v0 (np.ndarray): Starting vector
    #         v1 (np.ndarray): Final vector
    #         DOT_THRESHOLD (float): Threshold for considering the two vectors as
    #                                 colineal. Not recommended to alter this.
    #     Returns:
    #         v2 (np.ndarray): Interpolation vector between v0 and v1
    #     '''
    #     is_tensor = False
    #     if not isinstance(v0,np.ndarray):
    #         is_tensor = True
    #         device = v0.device
    #         v0 = v0.detach().cpu().numpy()
    #     if not isinstance(v1,np.ndarray):
    #         is_tensor = True
    #         device = v1.device  # overwrite if v0 is also Tensor
    #         v1 = v1.detach().cpu().numpy()
    #     # Copy the vectors to reuse them later
    #     v0_copy = np.copy(v0)
    #     v1_copy = np.copy(v1)
    #     # Normalize the vectors to get the directions and angles
    #     v0 = v0 / np.linalg.norm(v0)
    #     v1 = v1 / np.linalg.norm(v1)
    #     # Dot product with the normalized vectors (can't use np.dot in W)
    #     dot = np.sum(v0 * v1)
    #     # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    #     if np.abs(dot) > DOT_THRESHOLD:
    #         return lerp(t, v0_copy, v1_copy)
    #     # Calculate initial angle between v0 and v1
    #     theta_0 = np.arccos(dot)
    #     sin_theta_0 = np.sin(theta_0)
    #     # Angle at timestep t
    #     theta_t = theta_0 * t
    #     sin_theta_t = np.sin(theta_t)
    #     s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    #     s1 = sin_theta_t / sin_theta_0
    #     v2 = s0*v0_copy + s1*v1_copy
    if is_tensor:
        res = torch.from_numpy(v2).to(device)
    else:
        res = v2
    return res


def normalize_along_channel(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


# def extract_and_fill(spectrum, a, b, sr, hop_length):
#     """
#     Extract a 1-second segment from (a, b) and fill the rest of the segment using repeat or reflection.

#     Parameters:
#     spectrum (Tensor): The input spectrum tensor.
#     a (float): The start time of the region with energy.
#     b (float): The end time of the region with energy.
#     sr (int): The sample rate of the spectrum.
#     hop_length (int)

#     Returns:
#     Tensor: The processed spectrum tensor.
#     """
#     n_frames = spectrum.size(1)
#     n_frames_per_sec = sr // hop_length
#     mask = (spectrum!=0).float()
#     # Convert time to samples
#     a_frame = math.floor(a * sr / hop_length)
#     b_frame = math.ceil(b * sr / hop_length)
#     assert a_frame < n_frames and b_frame < n_frames
#     duration = b_frame - a_frame

#     # If the energy region is shorter than 1 second, adjust
#     extract_duration = duration // 2 if duration <= n_frames_per_sec else n_frames_per_sec
#     padding = duration - extract_duration

#     start_frame = random.randint(a_frame, b_frame-extract_duration)
#     segment = spectrum[:, start_frame:start_frame+extract_duration, :]
#     segment = segment.repeat(1, n_frames//extract_duration+1, 1)[:, :n_frames, :]
#     segment *= mask

#     return segment


def extract_and_fill(spec, stt_frame, end_frame, tgt_length):
    """
    Extract a region with <= `tgt_length` from (`stt_frame`, `end_frame`) and fill the rest of the spec by repeating the extracted region.

    Param:
        spec: Tensor: input spectrogram, shape = (C,T,F).
        a: float: The start time of the region with energy.
        b: float: The end time of the region with energy.


    Returns:
        Tensor: the processed spectrum tensor.
    """
    assert (spec.ndim == 3 or spec.ndim == 4), "Format the input `spec` with the shape = (C, T, F) or (B,C,T,F)."

    total_length = spec.size(-2)
    assert stt_frame < total_length and end_frame < total_length

    duration = end_frame - stt_frame
    mask = (spec != 0).float()

    # If the energy region is shorter than 1 second, adjust
    extract_duration = duration // 2 if duration <= tgt_length else tgt_length

    start_frame = random.randint(stt_frame, end_frame - extract_duration)

    if spec.ndim == 3:
        segment = spec[:, start_frame : start_frame + extract_duration, :]
        segment = segment.repeat(1, total_length // extract_duration + 1, 1)[
            :, :total_length, :
        ]
    else:
        segment = spec[:, :, start_frame : start_frame + extract_duration, :]
        segment = segment.repeat(1, 1, total_length // extract_duration + 1, 1)[
            :, :, :total_length, :
        ]

    segment *= mask
    return segment


def fill_with_neighbor(spec, stt_frame, end_frame, neighbor_length):
    """
    Fill a region from (`stt_frame`, `end_frame`) with neighbor of `neighbor_length`

    Param:
        spec: Tensor: input spectrogram, shape = (C,T,F).
        stt_frame: int: The start frame of the region with energy.
        end_frame: int: The end frame of the region with energy.
        neighbor_length: int: selected length of neighbor


    Returns:
        Tensor: the processed spectrum tensor.
    """
    assert spec.ndim == 3, "Format the input `spec` with the shape = (C, T, F)."

    total_length = spec.size(1)
    assert stt_frame < total_length and end_frame < total_length

    duration = end_frame - stt_frame
    mask = torch.zeros_like(spec)
    mask[:, stt_frame : end_frame + 1, :] = 1

    left_duration = min(math.ceil(neighbor_length / 2), stt_frame)
    right_duration = min(neighbor_length - left_duration, total_length - end_frame - 1)

    if left_duration + right_duration < 1:
        print("Warning: cannot find effect positive part!")
        return torch.randn_like(segment)

    left_segment = spec[:, stt_frame - left_duration : stt_frame, :]
    right_segment = spec[:, end_frame + 1 : end_frame + right_duration + 1, :]
    segment = torch.cat([left_segment, right_segment], dim=1)
    segment = segment.repeat(
        1, total_length // (left_duration + right_duration) + 1, 1
    )[:, :total_length, :]
    segment = segment * mask + spec * (1 - mask)

    return segment


# def slerp(t, A, B, eps=1e-8):
#     """
#     Spherical Linear Interpolation (SLERP) between points A and B on a sphere.
#     """
#     A = A / (torch.norm(A, p=2) + eps)
#     B = B / (torch.norm(B, p=2) + eps)

#     dot_product = torch.sum(A * B)
#     dot_product = torch.clamp(dot_product, -1.0, 1.0)

#     theta = torch.acos(dot_product)

#     if torch.abs(theta) < 1e-10:
#         return (1 - t) * A + t * B

#     sin_theta = torch.sin(theta)
#     A_factor = torch.sin((1 - t) * theta) / sin_theta
#     B_factor = torch.sin(t * theta) / sin_theta

#     return A_factor * A + B_factor * B


def lerp(t, v0, v1):
    """
    Linear interpolation in PyTorch.
    Args:
        t (float/torch.Tensor): Float value between 0.0 and 1.0
        v0 (torch.Tensor): Starting vector
        v1 (torch.Tensor): Final vector
    Returns:
        v2 (torch.Tensor): Interpolation vector between v0 and v1
    """
    return (1 - t) * v0 + t * v1


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """
    Spherical linear interpolation in PyTorch.
    Args:
        t (float/torch.Tensor): Float value between 0.0 and 1.0
        v0 (torch.Tensor): Starting vector
        v1 (torch.Tensor): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as collinear. Not recommended to alter this.
    Returns:
        v2 (torch.Tensor): Interpolation vector between v0 and v1
    """
    device = v0.device
    # Normalize the vectors to get the directions and angles
    v0_norm = v0 / torch.norm(v0)
    v1_norm = v1 / torch.norm(v1)
    # Dot product with the normalized vectors
    dot = torch.sum(v0_norm * v1_norm)
    # If absolute value of dot product is almost 1, vectors are ~collinear, so use lerp
    if torch.abs(dot) > DOT_THRESHOLD:
        return lerp(t, v0, v1)

    # Calculate initial angle between v0 and v1
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0 + s1 * v1

    return v2


def geodesic_distance(X, Y):
    """
    Compute the geodesic distance between two points X and Y on a sphere.
    """
    dot_product = torch.sum(X * Y)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    return torch.acos(dot_product)


def optimize_neighborhood_points(
    A,
    B,
    M,
    t,
    learning_rate=1e-4,
    iterations=100,
    enable_penalty=False,
    enable_tangent_proj=True,
):
    """
    Optimize the neighborhood points A_e and B_e to minimize the distance between
    the SLERP interpolation and the given interpolation point M.
    """
    # Initialize perturbations
    epsilon_A = torch.zeros_like(A, requires_grad=True)
    epsilon_B = torch.zeros_like(B, requires_grad=True)

    optimizer = torch.optim.SGD([epsilon_A, epsilon_B], lr=learning_rate)  # Adam

    for i in range(iterations):
        optimizer.zero_grad()

        # Compute current neighborhood points
        A_e = A + epsilon_A
        B_e = B + epsilon_B

        # Compute the SLERP interpolation
        P = slerp(t, A_e, B_e)

        # Compute the distance
        dist = geodesic_distance(M, P)
        if enable_penalty:
            orthogonality_penalty = torch.sum(A_e * B_e) ** 2
            dist += orthogonality_penalty

        # Backpropagation
        dist.backward()

        if enable_tangent_proj:
            with torch.no_grad():
                epsilon_A.grad = project_onto_tangent_space(epsilon_A.grad, A_e)
                epsilon_B.grad = project_onto_tangent_space(epsilon_B.grad, B_e)

        # Clip gradients to prevent large updates
        torch.nn.utils.clip_grad_norm_([epsilon_A, epsilon_B], max_norm=1.0)

        # Check gradients for NaNs
        if torch.isnan(epsilon_A.grad).any() or torch.isnan(epsilon_B.grad).any():
            print(f"NaN encountered in gradients at iteration {i}")
            break

        # Update perturbations
        optimizer.step()

    return A + epsilon_A.detach(), B + epsilon_B.detach()


# def optimize_neighborhood_points(A, B, M, t, learning_rate=1e-4, iterations=100, enable_penalty=False, eps=1e-8):
#     """
#     Optimize the neighborhood points A_e and B_e to minimize the distance between
#     the SLERP interpolation and the given interpolation point M.
#     """
#     # Initialize perturbations
#     epsilon_A = torch.zeros_like(A, requires_grad=True)
#     epsilon_B = torch.zeros_like(B, requires_grad=True)

#     optimizer = torch.optim.SGD([epsilon_A, epsilon_B], lr=learning_rate)

#     for _ in range(iterations):
#         optimizer.zero_grad()

#         # Compute current neighborhood points
#         A_e = A + epsilon_A
#         B_e = B + epsilon_B

#         # # Normalize to ensure they are on the unit sphere
#         # A_e = A_e / (torch.norm(A_e, p=2) + eps)
#         # B_e = B_e / (torch.norm(B_e, p=2) + eps)

#         # Compute the SLERP interpolation
#         P = slerp(t, A_e, B_e)

#         # Compute the distance
#         dist = geodesic_distance(M, P)
#         if enable_penalty:
#             orthogonality_penalty = torch.sum(A_e * B_e) ** 2
#             dist += orthogonality_penalty

#         # Backpropagation
#         dist.backward()

#         # Update perturbations
#         optimizer.step()

#     return A + epsilon_A.detach(), B + epsilon_B.detach()


# def optimize_neighborhood_points(A, B, M, t, learning_rate=2e-5, iterations=100, enable_penalty=False):
#     """
#     [Deprecated] this method tends to NaN
#     Optimize the neighborhood points A_e and B_e to minimize the distance between
#     the SLERP interpolation and the given interpolation point M.
#     """
#     # Initialize perturbations
#     A_e = A.clone().detach().requires_grad_(True)
#     B_e = B.clone().detach().requires_grad_(True)

#     optimizer = torch.optim.SGD([A_e, B_e], lr=learning_rate)

#     for _ in range(iterations):
#         optimizer.zero_grad()
#         # Compute the SLERP interpolation
#         P = slerp(t, A_e, B_e)

#         # Compute the distance
#         dist = geodesic_distance(M, P)
#         if enable_penalty:
#             orthogonality_penalty = torch.sum(A_e * B_e) ** 2
#             dist += orthogonality_penalty

#         # Backpropagation
#         dist.backward()

#         with torch.no_grad():
#             A_e.grad = project_onto_tangent_space(A_e.grad, A_e)
#             B_e.grad = project_onto_tangent_space(B_e.grad, B_e)

#         # Update perturbations
#         optimizer.step()

#     return A_e.detach().requires_grad_(False), B_e.detach().requires_grad_(False)


def project_onto_tangent_space(g, h, eps=1e-8):
    """
    Projects vector g onto the tangent space of vector h.

    Args:
        g (torch.Tensor): The vector to be projected.
        h (torch.Tensor): The vector whose tangent space g is projected onto.

    Returns:
        torch.Tensor: The projection of g onto the tangent space of h.
    """
    g = torch.tensor(g)
    h = torch.tensor(h)

    # Compute the dot product g . h
    dot_product = torch.sum(g * h)

    # Compute the squared norm of h, h . h
    h_norm_squared = torch.sum(h * h) + eps

    # Calculate the projection scalar
    proj_scalar = dot_product / h_norm_squared

    # Compute the component of g in the direction of h
    g_para = proj_scalar * h

    # Compute the projection of g onto the tangent space of h
    g_ortho = g - g_para

    return g_ortho


def label2caption(label, background_sound=None, template="{} can be heard"):
    r"""This is a helper function converting list of labels to captions."""
    if background_sound is None:
        return [template.format(", ".join(l)) for l in label]

    if isinstance(background_sound, str):
        background_sound = [[background_sound]] * len(label)

    assert len(label) == len(
        background_sound
    ), "the number of `background_sound` should match the number of `label`."

    caption = []
    for l, bg in zip(label, background_sound):
        cap = template.format(", ".join(l))
        cap += " with the background sounds of {}".format(", ".join(bg))
        caption.append(cap)

    return caption


def load_json(fname):
    with open(fname, "r") as f:
        data = json.load(f)
        return data


def identity_projection(g, *args, **kwargs):
    return g


def convert_float_to_int(data):
    data *= 32768
    data = np.nan_to_num(data, nan=0.0, posinf=32767, neginf=-32768)
    data = np.clip(data, -32768, 32767)
    return data


def get_edit_mask(mask, dx, dy, resize_scale_x, resize_scale_y):
    _mask = (
        F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            (
                int(mask.shape[-2] * resize_scale_y),
                int(mask.shape[-1] * resize_scale_x),
            ),
        )
        > 0.5
    )
    _mask = torch.roll(
        _mask,
        (int(dy * resize_scale_y), int(dx * resize_scale_x)),
        (-2, -1),
    )

    if resize_scale_x != 1 or resize_scale_y != 1:
        mask_res = torch.zeros(1, 1, mask.shape[-2], mask.shape[-1]).to(mask.device)
        pad_x = (mask_res.shape[-1] - _mask.shape[-1]) // 2
        pad_y = (mask_res.shape[-2] - _mask.shape[-2]) // 2
        px_tmp, py_tmp = max(pad_x, 0), max(pad_y, 0)
        px_tar, py_tar = max(-pad_x, 0), max(-pad_y, 0)
        mask_res[:,:,py_tmp:py_tmp+_mask.shape[-2],px_tmp:px_tmp+_mask.shape[-1]] = _mask[
            :,:,py_tar:py_tar+mask_res.shape[-2],px_tar:px_tar+mask_res.shape[-1]]
        # # Binary mask
        # mask_res = mask_res > 0.5
    # else:
    #     mask_res = _mask > 0.5
    else:
        mask_res = _mask

    return mask_res.squeeze()  # (y,x)


if __name__ == "__main__":
    # import torch
    # spec = torch.rand(1024, 64)
    # # import ipdb; ipdb.set_trace()
    # plot_spectrogram(spec.permute(1,0),'test.png')

    # m = torch.rand(4,4)
    # mask = [[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]]
    # mask = torch.tensor(mask).bool()
    # print(m)
    # print(get_neibor_with_mask(m, mask))

    # audio = torch.zeros(1,1024,64)
    # audio[:,250:750,:]=torch.rand(1,500,64)
    # res=extract_and_fill(audio, a=5,b=7.5, sr=16000, hop_length=160)
    # # import ipdb; ipdb.set_trace()

    # Try SLERP
    # A = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)  # Point on the unit sphere
    # B = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)  # Another point on the unit sphere

    # t = 0.5  # Interpolation parameter (0 <= t <= 1)
    # M = torch.tensor([0.7, 0.0, 1.2, 0.0], dtype=torch.float32) # slerp(t, A, B)  # Given interpolation point
    # A_e, B_e = optimize_neighborhood_points(A, B, M, t, enable_penalty=True)

    # print("Optimized A_e:", A_e)
    # print("Optimized B_e:", B_e)

    # spec = torch.arange(36).view(6,6)[None,...]
    # res = fill_with_neighbor(spec, 2, 4, 2)

    # Example usage
    g = torch.tensor([1.0, 2.0, 3.0])
    h = torch.tensor([4.0, 5.0, 6.0])

    g_ortho = project_onto_tangent_space(g, h)
    print(g_ortho)
    import ipdb

    ipdb.set_trace()
