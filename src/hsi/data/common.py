# src/hsi/data/common.py
from pathlib import Path
import numpy as np
from tifffile import imread

def to_hwb(x):
    if x.ndim == 3 and x.shape[0] < x.shape[-1]:  # (B,H,W) -> (H,W,B)
        x = np.moveaxis(x, 0, -1)
    return x

def load_tif(path: Path):
    arr = imread(str(path))
    return to_hwb(arr)

def maybe_rescale_reflectance(x: np.ndarray):
    x = x.astype(np.float32, copy=False)
    if np.nanmax(x) > 10:  # likely 0–10000 ints
        x /= 10000.0
    return x

def labels_rgb_to_ids(labels_rgb: np.ndarray):
    assert labels_rgb.ndim == 3 and labels_rgb.shape[-1] == 3, "expected (H,W,3) RGB labels"
    H, W, _ = labels_rgb.shape
    lr = labels_rgb[..., 0].astype(np.uint32)
    lg = labels_rgb[..., 1].astype(np.uint32)
    lb = labels_rgb[..., 2].astype(np.uint32)
    packed = (lr << 16) | (lg << 8) | lb
    flat = packed.reshape(-1)
    uniq, inv = np.unique(flat, return_inverse=True)
    ids = inv.reshape(H, W).astype(np.int32)
    palette = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.uint8)
    return ids, palette
