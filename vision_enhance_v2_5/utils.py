"""
Utility functions for the Vision Enhancement Engine v2.5.
Handles tensor conversions, padding, autocast, and image helpers.
"""

import math
from contextlib import contextmanager
from typing import Callable, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tensor ↔ NumPy conversions
# ---------------------------------------------------------------------------

def bgr_to_rgb_tensor(
    img: np.ndarray,
    device: torch.device = torch.device("cpu"),
    normalize: bool = True,
) -> torch.Tensor:
    """Convert a BGR uint8 numpy image to an RGB float32 tensor (NCHW).

    Args:
        img: HWC BGR numpy array, dtype uint8 or float32.
        device: Target torch device.
        normalize: If True, scale pixel values to [0, 1].

    Returns:
        Tensor of shape (1, 3, H, W) on the specified device.
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float32)
    if normalize and img.max() > 1.0:
        img = img / 255.0
    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # HWC → CHW → NCHW
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor.to(device)


def rgb_tensor_to_bgr(
    tensor: torch.Tensor,
    denormalize: bool = True,
) -> np.ndarray:
    """Convert an RGB float32 tensor (NCHW) back to a BGR uint8 numpy image.

    Args:
        tensor: Tensor of shape (1, 3, H, W), values in [0, 1].
        denormalize: If True, scale back to [0, 255].

    Returns:
        HWC BGR numpy array, dtype uint8.
    """
    img = tensor.squeeze(0).detach().cpu().clamp(0.0, 1.0)
    img = img.permute(1, 2, 0).numpy()  # CHW → HWC
    if denormalize:
        img = (img * 255.0).round().astype(np.uint8)
    # RGB → BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------

def pad_to_multiple(
    img: torch.Tensor,
    multiple: int,
    mode: str = "reflect",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad a NCHW tensor so that H and W are divisible by *multiple*.

    Returns:
        (padded_tensor, (pad_h, pad_w)) — the amount of padding added.
    """
    _, _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    padded = F.pad(img, (0, pad_w, 0, pad_h), mode=mode)
    return padded, (pad_h, pad_w)


def unpad(img: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
    """Remove padding added by :func:`pad_to_multiple`."""
    pad_h, pad_w = pad_hw
    if pad_h == 0 and pad_w == 0:
        return img
    h = img.shape[2] - pad_h
    w = img.shape[3] - pad_w
    return img[:, :, :h, :w]


# ---------------------------------------------------------------------------
# Autocast helper
# ---------------------------------------------------------------------------

@contextmanager
def autocast_context(device: torch.device):
    """Context manager for FP16 autocast on CUDA, identity on CPU.

    Usage::

        with autocast_context(device):
            output = model(input)
    """
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda" and torch.cuda.is_available():
        with torch.amp.autocast(device_type="cuda"):
            yield
    else:
        yield


# ---------------------------------------------------------------------------
# Image quality helpers
# ---------------------------------------------------------------------------

def resize_if_needed(
    img: np.ndarray,
    max_side: int = 1920,
) -> Tuple[np.ndarray, float]:
    """Downscale an image if its longest side exceeds *max_side*.

    Returns:
        (resized_image, scale_factor)
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img, 1.0
    scale = max_side / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


# ---------------------------------------------------------------------------
# Batch processing helper
# ---------------------------------------------------------------------------

def batch_frames(
    frames: list,
    batch_size: int = 4,
) -> list:
    """Split a list of frames into batches.

    Args:
        frames: List of numpy arrays.
        batch_size: Maximum frames per batch.

    Returns:
        List of lists, each containing up to batch_size frames.
    """
    return [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
