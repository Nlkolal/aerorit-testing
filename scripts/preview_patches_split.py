from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from hsi.data.aerorit import load_aerorit
from hsi.data.patches import create_patches, label_patches, remove_unlabeled_patches
from hsi.viz import make_rgb_from_cube

# ---------- settings ----------
ROOT = Path(r"C:\Users\Nikolai A\Documents\Thesis\data\aerorit")
PATCH_SIZE = 21
TRAIN_P, VAL_P = 0.30, 0.15
SEED = 42
SAVE_FIGS = False
OUT_DIR = Path("figures")
# ------------------------------


# --- load data and make RGB ---
refl, labels, palette = load_aerorit(ROOT)
rgb = make_rgb_from_cube(refl, (650, 550, 450))  # uint8 (H,W,3)
H, W = rgb.shape[:2]

# --- step 1: non-overlap patch grid ---
grid = create_patches(refl, PATCH_SIZE)            # shape (#rows, #cols)
split_grid = label_patches(grid, TRAIN_P, VAL_P, SEED)
split_grid = remove_unlabeled_patches(split_grid, labels, 3, PATCH_SIZE)

# --- build split overlay (train=green, val=orange, test=blue) ---
colors = {
    -1: np.array([77, 77, 77], dtype=np.uint8),
    0: np.array([0, 0, 0], dtype=np.uint8),        # unused
    1: np.array([0, 180, 80], dtype=np.uint8),     # train
    2: np.array([255, 165, 0], dtype=np.uint8),    # val
    3: np.array([70, 130, 180], dtype=np.uint8),   # test
}
# --- build overlay image (same as before) ---
overlay = np.zeros_like(rgb)  # (H,W,3) uint8
hdiv, wdiv = split_grid.shape
for i in range(hdiv):
    y = i * PATCH_SIZE
    for j in range(wdiv):
        x = j * PATCH_SIZE
        cls = int(split_grid[i, j])
        if cls != 0:
            overlay[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = colors[cls]

# --- counts for title ---
vals, counts = np.unique(split_grid, return_counts=True)
cnt = dict(zip(vals.tolist(), counts.tolist()))
n_tr, n_va, n_te = cnt.get(1, 0), cnt.get(2, 0), cnt.get(3, 0)

# --- make an RGBA overlay so only colored areas are semi-transparent ---
alpha = 0.35  # opacity of colored patches
H, W = rgb.shape[:2]
ov_rgba = np.zeros((H, W, 4), dtype=float)
mask = (overlay.sum(axis=-1) > 0)
ov_rgba[..., :3] = overlay.astype(float) / 255.0
ov_rgba[..., 3] = 0.0
ov_rgba[mask, 3] = alpha

# --- single plot: RGB + overlay ---
plt.figure(figsize=(8, 8))
plt.imshow(rgb)          # base
plt.imshow(ov_rgba)      # overlay with per-pixel alpha
plt.title(f"Patch split overlay  (train={n_tr}, val={n_va}, test={n_te})")
plt.axis("off")
plt.show()


