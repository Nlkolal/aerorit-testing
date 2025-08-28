from pathlib import Path
import numpy as np
from tifffile import imread

def to_hwb(x):
    # ensure (H, W, B) for cubes, (H, W) for labels
    if x.ndim == 3 and x.shape[0] < x.shape[-1]:  # (B,H,W) -> (H,W,B)
        x = np.moveaxis(x, 0, -1)
    return x

def load_tif(path):
    arr = imread(path)
    return to_hwb(arr)

def maybe_rescale_reflectance(x):
    x = x.astype(np.float32, copy=False)
    if np.issubdtype(x.dtype, np.integer) or np.nanmax(x) > 10:
        x /= 10000.0
    return x

def main():
    # EDIT these to your files
    root = Path(r"C:\Users\Nikolai A\Documents\Thesis\data")
    refl = load_tif(root / "image_hsi_reflectance.tif")
    labels = imread(root / "image_labels.tif")   # usually (H,W) ints

    # normalize reflectance if needed
    refl = maybe_rescale_reflectance(refl)

    print("Reflectance shape:", refl.shape)  # (H,W,B)
    print("Labels shape:", labels.shape)     # (H,W)
    H, W, B = refl.shape
    assert labels.shape == (H, W), "Label size must match cube size!"

    # quick label histogram
    ids, counts = np.unique(labels, return_counts=True)
    print("Label IDs:", ids.tolist())
    print("Counts   :", counts.tolist())

if __name__ == "__main__":
    main()
