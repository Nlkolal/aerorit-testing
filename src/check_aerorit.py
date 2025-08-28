from pathlib import Path
import numpy as np
from tifffile import imread

def to_hwb(x):
    if x.ndim == 3 and x.shape[0] < x.shape[-1]:  # (B,H,W) -> (H,W,B)
        x = np.moveaxis(x, 0, -1)
    return x

def load_tif(path):
    arr = imread(path)
    return to_hwb(arr)

def maybe_rescale_reflectance(x):
    x = x.astype(np.float32, copy=False)
    if np.nanmax(x) > 10:  # likely stored as 0–10000 ints
        x /= 10000.0
    return x

def labels_rgb_to_ids(labels_rgb: np.ndarray):
    """
    Convert (H,W,3) RGB color labels to (H,W) integer class IDs.
    Also returns the color palette array of shape (C,3) mapping id->RGB.
    """
    assert labels_rgb.ndim == 3 and labels_rgb.shape[-1] == 3, "expected (H,W,3) RGB labels"
    H, W, _ = labels_rgb.shape

    # Pack RGB into a single uint32: R<<16 | G<<8 | B (fast & memory-friendly)
    lr = labels_rgb[..., 0].astype(np.uint32)
    lg = labels_rgb[..., 1].astype(np.uint32)
    lb = labels_rgb[..., 2].astype(np.uint32)
    packed = (lr << 16) | (lg << 8) | lb
    flat = packed.reshape(-1)

    uniq, inv = np.unique(flat, return_inverse=True)  # uniq colors, and index per pixel
    ids = inv.reshape(H, W).astype(np.int32)

    # Reconstruct palette (id -> RGB)
    palette = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.uint8)
    return ids, palette

def main():
    root = Path(r"C:\Users\Nikolai A\Documents\Thesis\data")

    refl = load_tif(root / "image_hsi_reflectance.tif")   # (H,W,B)
    labels_raw = load_tif(root / "image_labels.tif")      # (H,W) or (H,W,3)

    refl = maybe_rescale_reflectance(refl)

    H, W, B = refl.shape
    print("Reflectance shape:", refl.shape)

    # If labels are RGB, convert to IDs
    if labels_raw.ndim == 3 and labels_raw.shape[-1] == 3:
        labels, palette = labels_rgb_to_ids(labels_raw)
        print("Labels were RGB; converted to IDs.")
        print("Palette (id -> [R,G,B]) shape:", palette.shape)
        # (Optional) save the palette so you know which color is which id
        np.save(root / "labels_palette.npy", palette)
    elif labels_raw.ndim == 2:
        labels = labels_raw.astype(np.int32, copy=False)
    else:
        raise ValueError(f"Unexpected labels shape: {labels_raw.shape}")

    assert labels.shape == (H, W), "Label size must match cube size!"
    print("Labels shape:", labels.shape)

    # Quick class histogram
    ids, counts = np.unique(labels, return_counts=True)
    print("Class IDs:", ids.tolist())
    print("Counts   :", counts.tolist())

if __name__ == "__main__":
    main()
