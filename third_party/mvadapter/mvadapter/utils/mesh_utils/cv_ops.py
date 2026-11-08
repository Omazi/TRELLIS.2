# Remplace la version CV-CUDA d'origine (cvcuda n'existe pas pour CUDA 13) par OpenCV / scipy, sur CPU.
# Mêmes signatures et mêmes conventions d'entrée/sortie que l'original.
from typing import Optional

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def _to_u8(x: torch.Tensor) -> np.ndarray:
    x = x.detach()
    if x.dtype != torch.uint8:
        x = (x.float().clamp(0, 1) * 255).round().to(torch.uint8)
    return x.cpu().numpy()


def inpaint_cvc(image: torch.Tensor, mask: torch.Tensor, padding_size: int, return_dtype: Optional[torch.dtype] = None):
    """Remplit les pixels où mask est vrai. Grandes zones (bourrage UV autour des îlots) : couleur du
    pixel valide le plus proche ; petites zones : inpainting Telea d'OpenCV (comme cvcuda.inpaint)."""
    input_dtype, device = image.dtype, image.device
    img = _to_u8(image)                                   # HWC
    m = mask.detach().cpu().numpy()
    m = (m > (127 if m.dtype == np.uint8 else 0.5)) if m.dtype != bool else m
    if m.mean() > 0.05 or not (~m).any():
        if (~m).any():
            idx = distance_transform_edt(m, return_distances=False, return_indices=True)
            out = img[idx[0], idx[1]]
        else:
            out = img
    else:
        out = cv2.inpaint(np.ascontiguousarray(img), m.astype(np.uint8) * 255, max(1, int(padding_size)), cv2.INPAINT_TELEA)
        if out.ndim == 2 and img.ndim == 3:
            out = out[..., None]
    out = torch.as_tensor(np.ascontiguousarray(out), device=device)
    if return_dtype == torch.uint8 or input_dtype == torch.uint8:
        return out
    return out.to(dtype=input_dtype) / 255.0


def batch_inpaint_cvc(images: torch.Tensor, masks: torch.Tensor, padding_size: int, return_dtype: Optional[torch.dtype] = None):
    return torch.stack([inpaint_cvc(i, m, padding_size, return_dtype) for i, m in zip(images, masks)], dim=0)


def _morph(masks: torch.Tensor, kernel_size: int, op, return_dtype):
    input_dtype, device = masks.dtype, masks.device
    m = _to_u8(masks if masks.dtype == torch.uint8 else (masks.float() > 0).float())
    k = np.ones((kernel_size, kernel_size), np.uint8)
    out = np.stack([op(x, k, borderType=cv2.BORDER_CONSTANT, borderValue=0) for x in m])
    out = torch.as_tensor(out, device=device)
    if return_dtype == torch.uint8 or input_dtype == torch.uint8:
        return out
    return (out > 0).to(dtype=input_dtype)


def batch_erode(masks: torch.Tensor, kernel_size: int, return_dtype: Optional[torch.dtype] = None):
    return _morph(masks, kernel_size, cv2.erode, return_dtype)


def batch_dilate(masks: torch.Tensor, kernel_size: int, return_dtype: Optional[torch.dtype] = None):
    return _morph(masks, kernel_size, cv2.dilate, return_dtype)
