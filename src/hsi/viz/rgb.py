# src/hsi/viz/rgb.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def wavelength_to_index(wl_nm: float, start_nm: int = 400, step_nm: int = 10, max_bands: int | None = None) -> int:
    """Map wavelength to band index (AeroRIT default: 400–900nm step 10nm)."""
    idx = int(round((wl_nm - start_nm) / step_nm))
    if max_bands is not None:
        idx = max(0, min(idx, max_bands - 1))
    return idx

def percentile_stretch(img: np.ndarray, p_low: float = 2, p_high: float = 98, per_channel: bool = True) -> np.ndarray:
    """Robust 0–1 stretch for display."""
    img = img.astype(np.float32, copy=False)
    if img.ndim == 3 and per_channel:
        out = np.empty_like(img, dtype=np.float32)
        for c in range(img.shape[-1]):
            lo, hi = np.percentile(img[..., c], [p_low, p_high])
            if hi <= lo: hi = lo + 1e-6
            out[..., c] = np.clip((img[..., c] - lo) / (hi - lo), 0, 1)
        return out
    lo, hi = np.percentile(img, [p_low, p_high])
    if hi <= lo: hi = lo + 1e-6
    return np.clip((img - lo) / (hi - lo), 0, 1)

def make_rgb_from_cube(cube: np.ndarray, rgb_wls=(650, 550, 450), start_nm=400, step_nm=10) -> np.ndarray:
    """Build an RGB image from a hyperspectral cube (H,W,B)."""
    B = cube.shape[-1]
    idx = [wavelength_to_index(w, start_nm, step_nm, B) for w in rgb_wls]
    rgb = np.stack([cube[..., idx[0]], cube[..., idx[1]], cube[..., idx[2]]], axis=-1)
    return percentile_stretch(rgb, 2, 98, per_channel=True)

def show_img(img: np.ndarray, title: str | None = None, downsample: int = 1, save_path: str | None = None):
    """Display an image (expects RGB float 0–1 or uint8)."""
    if downsample > 1:
        img = img[::downsample, ::downsample]
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    if title: plt.title(title)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

def show_band(cube: np.ndarray, band_index: int, title: str | None = None, downsample: int = 1, cmap: str = "gray", save_path: str | None = None):
    """Display a single band with percentile stretch."""
    img = percentile_stretch(cube[..., band_index], 2, 98, per_channel=False)
    if downsample > 1:
        img = img[::downsample, ::downsample]
    plt.figure(figsize=(7, 6))
    plt.imshow(img, cmap=cmap)
    if title is None: title = f"Band {band_index}"
    plt.title(title); plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
