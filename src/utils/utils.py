import cv2
import torch
import numpy as np
import torch.nn.functional as F


def resize_numpy_image(image, max_resolution=768 * 768, resize_short_edge=None):
    h, w = image.shape[:2]
    w_org = image.shape[1]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    scale = w / w_org
    return image, scale


def split_ldm(ldm):
    x = []
    y = []
    for p in ldm:
        x.append(p[0])
        y.append(p[1])
    return x, y


def process_move(
    path_mask, # target region of original map
    h,
    w,
    dx,
    dy,
    scale,
    input_scale,
    resize_scale_x,
    resize_scale_y,
    up_scale,
    up_ft_index,
    w_edit,
    w_content,
    w_contrast,
    w_inpaint,
    precision,
    path_mask_ref=None,
    path_mask_keep=None,
):
    dx, dy = dx * input_scale, dy * input_scale
    mask_x0 = path_mask
    mask_x0_ref = path_mask_ref
    mask_x0_keep = path_mask_keep

    mask_x0 = (mask_x0 > 0.5).float().to("cuda", dtype=precision)
    if mask_x0_ref is not None:
        mask_x0_ref = (mask_x0_ref > 0.5).float().to("cuda", dtype=precision)
    # Define region to keep if `path_mask_keep` is given
    if mask_x0_keep is not None:
        mask_x0_keep = (mask_x0_keep > 0.5).float().to("cuda", dtype=precision)
        mask_keep = (
            F.interpolate(
                mask_x0_keep[None, None],
                (int(mask_x0_keep.shape[-2] // scale), int(mask_x0_keep.shape[-1] // scale)),
            )
            > 0.5
        ).float()
    else:
        mask_keep = None
        
    mask_org = (
        F.interpolate(
            mask_x0[None, None],
            (int(mask_x0.shape[-2] // scale), int(mask_x0.shape[-1] // scale)),
        )
        > 0.5
    )

    mask_tar = (
        F.interpolate(
            mask_x0[None, None],
            (
                int(mask_x0.shape[-2] // scale * resize_scale_y),
                int(mask_x0.shape[-1] // scale * resize_scale_x),
            ),
        )
        > 0.5
    )
    mask_cur = torch.roll(
        mask_tar,
        (int(dy // scale * resize_scale_y), int(dx // scale * resize_scale_x)),
        (-2, -1),
    )

    temp = torch.zeros(1, 1, mask_org.shape[-2], mask_org.shape[-1]).to(
        mask_org.device
        )
    pad_x = (temp.shape[-1] - mask_cur.shape[-1]) // 2
    pad_y = (temp.shape[-2] - mask_cur.shape[-2]) // 2
    px_tmp, py_tmp = max(pad_x, 0), max(pad_y, 0)
    px_tar, py_tar = max(-pad_x, 0), max(-pad_y, 0)
    temp[:,:,py_tmp:py_tmp+mask_cur.shape[-2],px_tmp:px_tmp+mask_cur.shape[-1]] = mask_cur[
        :,:,py_tar:py_tar+temp.shape[-2],px_tar:px_tar+temp.shape[-1]]
    # To avoid mask misaligned by shifting and cropping
    _mask_valid = torch.zeros_like(mask_cur)
    _mask_valid[:,:,py_tar:py_tar+temp.shape[-2],px_tar:px_tar+temp.shape[-1]] = 1
    _mask_valid = (torch.roll(
        _mask_valid,
        (-int(dy // scale * resize_scale_y), int(-dx // scale * resize_scale_x)),
        (-2, -1),
    ) > 0.5)
    mask_tar = torch.logical_and(mask_tar, _mask_valid)

    # Ensure the editing region is within the spectrogram
    if resize_scale_x > 1 or resize_scale_y > 1:
        sum_before = torch.sum(mask_tar) # replace `mask_cur` here
        sum_after = torch.sum(temp)
        if sum_after != sum_before:
            raise ValueError("Resize out of bounds, exiting.")
        
    mask_cur = temp > 0.5

    # Region of uninterested region is selected region when `mask_keep` is given
    if mask_keep is not None:
        mask_other = mask_keep > 0.5
    else:
        mask_other = (1 - ((mask_cur + mask_org) > 0.5).float()) > 0.5
    mask_overlap = ((mask_cur.float() + mask_org.float()) > 1.5).float()
    mask_non_overlap = (mask_org.float() - mask_overlap) > 0.5

    return {
        "mask_x0": mask_x0,
        "mask_x0_ref": mask_x0_ref,
        "mask_x0_keep": mask_x0_keep,
        "mask_tar": mask_tar,
        "mask_cur": mask_cur,
        "mask_other": mask_other,
        "mask_overlap": mask_overlap,
        "mask_non_overlap": mask_non_overlap,
        "mask_keep": mask_keep,
        "up_scale": up_scale,
        "up_ft_index": up_ft_index,
        "resize_scale_x": resize_scale_x,
        "resize_scale_y": resize_scale_y,
        "w_edit": w_edit,
        "w_content": w_content,
        "w_contrast": w_contrast,
        "w_inpaint": w_inpaint,
    }


def process_paste(
    path_mask,
    h,
    w,
    dx,
    dy,
    scale,
    input_scale,
    up_scale,
    up_ft_index,
    w_edit,
    w_content,
    precision,
    resize_scale_x,
    resize_scale_y,
):
    dx, dy = dx * input_scale, dy * input_scale
    if isinstance(path_mask, str):
        mask_base = cv2.imread(path_mask)
    else:
        mask_base = path_mask

    mask_base = mask_base[None, None]
    dict_mask = {}

    mask_base = (mask_base > 0.5).to("cuda", dtype=precision)

    #####[START] Original rescale and fit method.#####
    # if resize_scale is not None and resize_scale != 1:
    #     hi, wi = mask_base.shape[-2], mask_base.shape[-1]
    #     mask_base = F.interpolate(
    #         mask_base, (int(hi * resize_scale), int(wi * resize_scale))
    #     )
    #     pad_size_x = np.abs(mask_base.shape[-1] - wi) // 2
    #     pad_size_y = np.abs(mask_base.shape[-2] - hi) // 2
    #     if resize_scale > 1:
    #         mask_base = mask_base[
    #             :, :, pad_size_y : pad_size_y + hi, pad_size_x : pad_size_x + wi
    #         ]
    #     else:
    #         temp = torch.zeros(1, 1, hi, wi).to(mask_base.device)
    #         temp[
    #             :,
    #             :,
    #             pad_size_y : pad_size_y + mask_base.shape[-2],
    #             pad_size_x : pad_size_x + mask_base.shape[-1],
    #         ] = mask_base
    #         mask_base = temp
    #####[END] Original rescale and fit method.#####

    hi, wi = mask_base.shape[-2], mask_base.shape[-1]
    mask_base = F.interpolate(
        mask_base, (int(hi*resize_scale_y), int(wi*resize_scale_x))
        )
    temp = torch.zeros(1, 1, hi, wi).to(mask_base.device)
    pad_x, pad_y = (wi - mask_base.shape[-1]) // 2, (hi - mask_base.shape[-2]) // 2
    px_tmp, py_tmp = max(pad_x, 0), max(pad_y, 0)
    px_tar, py_tar = max(-pad_x, 0), max(-pad_y, 0)
    temp[:,:,py_tmp:py_tmp+mask_base.shape[-2],px_tmp:px_tmp+mask_base.shape[-1]] = mask_base[
        :,:,py_tar:py_tar+temp.shape[-2],px_tar:px_tar+temp.shape[-1]]
    mask_base = temp

    mask_replace = mask_base.clone()
    mask_base = torch.roll(
        mask_base, (int(dy*resize_scale_y), int(dx*resize_scale_x)), (-2, -1))  # (C,T,F)
    dict_mask["base"] = mask_base[0, 0]
    dict_mask["replace"] = mask_replace[0, 0]
    mask_replace = (mask_replace > 0.5).to("cuda", dtype=precision)

    mask_base_cur = (
        F.interpolate(
            mask_base,
            (int(mask_base.shape[-2]//scale), int(mask_base.shape[-1]//scale)),
        )
        > 0.5
    )
    mask_replace_cur = torch.roll(
        mask_base_cur, (-int(dy/scale), -int(dx/scale)), (-2, -1)
        )

    return {
        "dict_mask": dict_mask,
        "mask_base_cur": mask_base_cur,
        "mask_replace_cur": mask_replace_cur,
        "up_scale": up_scale,
        "up_ft_index": up_ft_index,
        "w_edit": w_edit,
        "w_content": w_content,
        "w_edit": w_edit,
        "w_content": w_content,
    }


# def process_paste(
#     path_mask,
#     h,
#     w,
#     dx,
#     dy,
#     scale,
#     input_scale,
#     up_scale,
#     up_ft_index,
#     w_edit,
#     w_content,
#     precision,
#     resize_scale=None,
# ):
#     dx, dy = dx * input_scale, dy * input_scale
#     if isinstance(path_mask, str):
#         mask_base = cv2.imread(path_mask)
#     else:
#         mask_base = path_mask

#     mask_base = mask_base[None, None]
#     dict_mask = {}

#     mask_base = (mask_base > 0.5).to("cuda", dtype=precision)
#     if resize_scale is not None and resize_scale != 1:
#         hi, wi = mask_base.shape[-2], mask_base.shape[-1]
#         mask_base = F.interpolate(
#             mask_base, (int(hi * resize_scale), int(wi * resize_scale))
#         )
#         pad_size_x = np.abs(mask_base.shape[-1] - wi) // 2
#         pad_size_y = np.abs(mask_base.shape[-2] - hi) // 2
#         if resize_scale > 1:
#             mask_base = mask_base[
#                 :, :, pad_size_y : pad_size_y + hi, pad_size_x : pad_size_x + wi
#             ]
#         else:
#             temp = torch.zeros(1, 1, hi, wi).to(mask_base.device)
#             temp[
#                 :,
#                 :,
#                 pad_size_y : pad_size_y + mask_base.shape[-2],
#                 pad_size_x : pad_size_x + mask_base.shape[-1],
#             ] = mask_base
#             mask_base = temp
#     mask_replace = mask_base.clone()
#     mask_base = torch.roll(mask_base, (int(dy), int(dx)), (-2, -1))  # (C,T,F)
#     dict_mask["base"] = mask_base[0, 0]
#     dict_mask["replace"] = mask_replace[0, 0]
#     mask_replace = (mask_replace > 0.5).to("cuda", dtype=precision)

#     mask_base_cur = (
#         F.interpolate(
#             mask_base,
#             (int(mask_base.shape[-2] // scale), int(mask_base.shape[-1] // scale)),
#         )
#         > 0.5
#     )
#     mask_replace_cur = torch.roll(
#         mask_base_cur, (-int(dy / scale), -int(dx / scale)), (-2, -1)
#     )

#     return {
#         "dict_mask": dict_mask,
#         "mask_base_cur": mask_base_cur,
#         "mask_replace_cur": mask_replace_cur,
#         "up_scale": up_scale,
#         "up_ft_index": up_ft_index,
#         "w_edit": w_edit,
#         "w_content": w_content,
#         "w_edit": w_edit,
#         "w_content": w_content,
#     }


def process_remove(
    path_mask,
    h,
    w,
    dx,
    dy,
    scale,
    input_scale,
    up_scale,
    up_ft_index,
    w_edit,
    w_contrast,
    w_content,
    precision,
    resize_scale_x,
    resize_scale_y,
):
    dx, dy = dx * input_scale, dy * input_scale
    if isinstance(path_mask, str):
        mask_base = cv2.imread(path_mask)
    else:
        mask_base = path_mask

    mask_base = mask_base[None, None]
    dict_mask = {}

    mask_base = (mask_base > 0.5).to("cuda", dtype=precision)
    #####[START] Original rescale and fit method.#####
    # if resize_scale is not None and resize_scale != 1:
    #     hi, wi = mask_base.shape[-2], mask_base.shape[-1]
    #     mask_base = F.interpolate(
    #         mask_base, (int(hi * resize_scale), int(wi * resize_scale))
    #     )
    #     pad_size_x = np.abs(mask_base.shape[-1] - wi) // 2
    #     pad_size_y = np.abs(mask_base.shape[-2] - hi) // 2
    #     if resize_scale > 1:
    #         mask_base = mask_base[
    #             :, :, pad_size_y : pad_size_y + hi, pad_size_x : pad_size_x + wi
    #         ]
    #     else:
    #         temp = torch.zeros(1, 1, hi, wi).to(mask_base.device)
    #         temp[
    #             :,
    #             :,
    #             pad_size_y : pad_size_y + mask_base.shape[-2],
    #             pad_size_x : pad_size_x + mask_base.shape[-1],
    #         ] = mask_base
    #         mask_base = temp
    #####[END] Original rescale and fit method.#####
    hi, wi = mask_base.shape[-2], mask_base.shape[-1]
    mask_base = F.interpolate(
        mask_base, (int(hi*resize_scale_y), int(wi*resize_scale_x))
        )
    temp = torch.zeros(1, 1, hi, wi).to(mask_base.device)
    pad_x, pad_y = (wi - mask_base.shape[-1]) // 2, (hi - mask_base.shape[-2]) // 2
    px_tmp, py_tmp = max(pad_x, 0), max(pad_y, 0)
    px_tar, py_tar = max(-pad_x, 0), max(-pad_y, 0)
    temp[:,:,py_tmp:py_tmp+mask_base.shape[-2],px_tmp:px_tmp+mask_base.shape[-1]] = mask_base[
        :,:,py_tar:py_tar+temp.shape[-2],px_tar:px_tar+temp.shape[-1]]
    mask_base = temp

    mask_replace = mask_base.clone()
    mask_base = torch.roll(mask_base, (int(dy), int(dx)), (-2, -1))  # (C,T,F)
    dict_mask["base"] = mask_base[0, 0]
    dict_mask["replace"] = mask_replace[0, 0]
    mask_replace = (mask_replace > 0.5).to("cuda", dtype=precision)

    mask_base_cur = (
        F.interpolate(
            mask_base,
            (int(mask_base.shape[-2] // scale), int(mask_base.shape[-1] // scale)),
        )
        > 0.5
    )
    mask_replace_cur = torch.roll(
        mask_base_cur, (-int(dy / scale), -int(dx / scale)), (-2, -1)
    )

    return {
        "dict_mask": dict_mask,
        "mask_base_cur": mask_base_cur,
        "mask_replace_cur": mask_replace_cur,
        "up_scale": up_scale,
        "up_ft_index": up_ft_index,
        "w_edit": w_edit,
        "w_contrast": w_contrast,
        "w_content": w_content,
    }


# def process_remove(
#     path_mask,
#     h,
#     w,
#     dx,
#     dy,
#     scale,
#     input_scale,
#     up_scale,
#     up_ft_index,
#     w_edit,
#     w_contrast,
#     w_content,
#     precision,
#     resize_scale=None,
# ):
#     dx, dy = dx * input_scale, dy * input_scale
#     if isinstance(path_mask, str):
#         mask_base = cv2.imread(path_mask)
#     else:
#         mask_base = path_mask

#     mask_base = mask_base[None, None]
#     dict_mask = {}

#     mask_base = (mask_base > 0.5).to("cuda", dtype=precision)
#     if resize_scale is not None and resize_scale != 1:
#         hi, wi = mask_base.shape[-2], mask_base.shape[-1]
#         mask_base = F.interpolate(
#             mask_base, (int(hi * resize_scale), int(wi * resize_scale))
#         )
#         pad_size_x = np.abs(mask_base.shape[-1] - wi) // 2
#         pad_size_y = np.abs(mask_base.shape[-2] - hi) // 2
#         if resize_scale > 1:
#             mask_base = mask_base[
#                 :, :, pad_size_y : pad_size_y + hi, pad_size_x : pad_size_x + wi
#             ]
#         else:
#             temp = torch.zeros(1, 1, hi, wi).to(mask_base.device)
#             temp[
#                 :,
#                 :,
#                 pad_size_y : pad_size_y + mask_base.shape[-2],
#                 pad_size_x : pad_size_x + mask_base.shape[-1],
#             ] = mask_base
#             mask_base = temp
#     mask_replace = mask_base.clone()
#     mask_base = torch.roll(mask_base, (int(dy), int(dx)), (-2, -1))  # (C,T,F)
#     dict_mask["base"] = mask_base[0, 0]
#     dict_mask["replace"] = mask_replace[0, 0]
#     mask_replace = (mask_replace > 0.5).to("cuda", dtype=precision)

#     mask_base_cur = (
#         F.interpolate(
#             mask_base,
#             (int(mask_base.shape[-2] // scale), int(mask_base.shape[-1] // scale)),
#         )
#         > 0.5
#     )
#     mask_replace_cur = torch.roll(
#         mask_base_cur, (-int(dy / scale), -int(dx / scale)), (-2, -1)
#     )

#     return {
#         "dict_mask": dict_mask,
#         "mask_base_cur": mask_base_cur,
#         "mask_replace_cur": mask_replace_cur,
#         "up_scale": up_scale,
#         "up_ft_index": up_ft_index,
#         "w_edit": w_edit,
#         "w_contrast": w_contrast,
#         "w_content": w_content,
#     }