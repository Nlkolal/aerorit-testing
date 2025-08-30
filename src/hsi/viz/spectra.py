# src/hsi/viz/spectra.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Tuple, Sequence

# ---- helpers ---------------------------------------------------------------

def nm_axis(
    B: int,
    start_nm: float = 400.0,
    step_nm: float = 10.0,
    band_wavelengths: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    """Return wavelength (nm) for each band index 0..B-1."""
    if band_wavelengths is not None:
        wl = np.asarray(band_wavelengths, dtype=float)
        if wl.shape[0] != B:
            raise ValueError(f"band_wavelengths has len {wl.shape[0]} but cube has {B} bands")
        return wl
    return start_nm + step_nm * np.arange(B, dtype=float)

def nm_to_thz(wl_nm: np.ndarray) -> np.ndarray:
    """Convert wavelength (nm) to frequency (THz). f[THz] = 299792.458 / wl[nm]."""
    return 299_792.458 / np.asarray(wl_nm, dtype=float)

def _validate_coords(H: int, W: int, coords: Iterable[Tuple[int,int]]) -> list[Tuple[int,int]]:
    cc = [(int(r), int(c)) for (r, c) in coords]
    for (r, c) in cc:
        if not (0 <= r < H and 0 <= c < W):
            raise ValueError(f"Coordinate {(r,c)} out of bounds for image {H}x{W}")
    return cc

# ---- line plots ------------------------------------------------------------

def plot_spectrum(
    cube: np.ndarray,
    row: int, col: int,
    mode: str = "nm",            # "nm" or "thz"
    start_nm: float = 400.0,
    step_nm: float = 10.0,
    band_wavelengths: np.ndarray | Sequence[float] | None = None,
    title: str | None = None,
    save_path: str | None = None,
    y_range: tuple[float, float] | None = None,
):
    """Plot a single pixel spectrum."""
    H, W, B = cube.shape
    x_nm = nm_axis(B, start_nm, step_nm, band_wavelengths)
    x = x_nm if mode.lower() == "nm" else nm_to_thz(x_nm)
    y = cube[row, col, :].astype(np.float32)

    plt.figure(figsize=(7, 4))
    plt.plot(x, y)
    plt.xlabel("Wavelength (nm)" if mode.lower() == "nm" else "Frequency (THz)")
    plt.ylabel("Reflectance")
    if title is None:
        title = f"Spectrum @ (r={row}, c={col})"
    if y_range is not None:                    
        plt.ylim(*y_range)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

def plot_spectra(
    cube: np.ndarray,
    coords: Iterable[Tuple[int,int]],
    labels: Sequence[str] | None = None,
    mode: str = "nm",            # "nm" or "thz"
    start_nm: float = 400.0,
    step_nm: float = 10.0,
    band_wavelengths: np.ndarray | Sequence[float] | None = None,
    title: str | None = None,
    save_path: str | None = None,
    y_range: tuple[float, float] | None = None,
):
    """Plot multiple pixel spectra on one chart."""
    H, W, B = cube.shape
    coords = _validate_coords(H, W, coords)
    x_nm = nm_axis(B, start_nm, step_nm, band_wavelengths)
    x = x_nm if mode.lower() == "nm" else nm_to_thz(x_nm)

    plt.figure(figsize=(8, 5))
    for i, (r, c) in enumerate(coords):
        y = cube[r, c, :].astype(np.float32)
        #lbl = labels[i] if labels and i < len(labels) else f"({r},{c})"
        plt.plot(x, y, label=None, alpha=0.9) #label=lbl
    plt.xlabel("Wavelength (nm)" if mode.lower() == "nm" else "Frequency (THz)")
    plt.ylabel("Reflectance")
    if y_range is not None:                      # <-- added
        plt.ylim(*y_range)
    if title is None:
        title = f"{len(coords)} spectra"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

# ---- aggregated “spectrogram-like” heatmap --------------------------------

def spectra_heatmap(
    cube: np.ndarray,
    coords: Iterable[Tuple[int,int]],
    mode: str = "nm",            # "nm" or "thz"
    start_nm: float = 400.0,
    step_nm: float = 10.0,
    band_wavelengths: np.ndarray | Sequence[float] | None = None,
    bins_axis: int | Sequence[float] | None = None,   # None -> use band centers
    bins_reflect: int | Sequence[float] = 100,
    reflect_range: Tuple[float,float] | None = None,
    title: str | None = None,
    save_path: str | None = None,
):
    """
    Aggregate selected pixels into a 2D histogram: spectral axis (x) vs reflectance (y).
    Think of it as a 'spectrogram-like' density view across all chosen pixels.
    """
    H, W, B = cube.shape
    coords = _validate_coords(H, W, coords)

    # X axis: nm or THz
    x_nm = nm_axis(B, start_nm, step_nm, band_wavelengths)
    x_band = x_nm if mode.lower() == "nm" else nm_to_thz(x_nm)

    # Build samples: for each coord, add (x_band[j], y[j]) for all bands j
    xs, ys = [], []
    for (r, c) in coords:
        y = cube[r, c, :].astype(np.float32)
        xs.append(np.tile(x_band, 1))  # shape (B,)
        ys.append(y)
    X = np.concatenate(xs)  # (len(coords)*B,)
    Y = np.concatenate(ys)

    # Binning
    if bins_axis is None:
        # use B bins aligned with band centers
        # ensure x bins cover the min..max with B bins
        x_min, x_max = float(x_band.min()), float(x_band.max())
        bins_axis = B
        x_range = (x_min, x_max)
    else:
        x_range = (float(np.min(x_band)), float(np.max(x_band)))

    if reflect_range is None:
        y_min, y_max = float(np.min(Y)), float(np.max(Y))
    else:
        y_min, y_max = reflect_range

    plt.figure(figsize=(8, 5))
    h = plt.hist2d(X, Y, bins=[bins_axis, bins_reflect], range=[x_range, (y_min, y_max)], cmap="viridis")
    plt.colorbar(h[3], label="Counts")
    plt.xlabel("Wavelength (nm)" if mode.lower() == "nm" else "Frequency (THz)")
    plt.ylabel("Reflectance")
    if title is None:
        title = f"Spectral density for {len(coords)} pixels"
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
