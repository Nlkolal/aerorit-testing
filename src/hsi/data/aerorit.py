# src/hsi/data/aerorit.py
from pathlib import Path
import numpy as np
from .common import load_tif, maybe_rescale_reflectance, labels_rgb_to_ids

ID_TO_NAME = {
    0:"road", 1:"vegetation", 2:"water", 3:"undefined", 4:"bulding", 5:"car",  # adjust to your palette
    -1:"ignore",  # if you have a no-data/edge class
}

def load_aerorit(
    root: str | Path,
    reflectance_name: str = "image_hsi_reflectance.tif",
    labels_name: str = "image_labels.tif",
    scale_reflectance: bool = True,
    save_palette: bool = False,
    palette_path: str | Path | None = None,
):
    """
    Returns:
        refl   : (H, W, B) float32 reflectance in [0,1] if scaled
        labels : (H, W) int32 class IDs
        palette: (C, 3) uint8 color table or None (if labels already IDs)
    """
    root = Path(root)

    refl = load_tif(root / reflectance_name)         # (H,W,B)
    if scale_reflectance:
        refl = maybe_rescale_reflectance(refl)

    labels_raw = load_tif(root / labels_name)        # (H,W) or (H,W,3)
    palette = None
    if labels_raw.ndim == 3 and labels_raw.shape[-1] == 3:
        labels, palette = labels_rgb_to_ids(labels_raw)
        if save_palette:
            p = Path(palette_path) if palette_path else (root / "labels_palette.npy")
            np.save(p, palette)
    elif labels_raw.ndim == 2:
        labels = labels_raw.astype(np.int32, copy=False)
    else:
        raise ValueError(f"Unexpected labels shape: {labels_raw.shape}")

    if labels.shape != refl.shape[:2]:
        raise ValueError(f"Label size {labels.shape} != cube size {refl.shape[:2]}")

    return refl, labels, palette
