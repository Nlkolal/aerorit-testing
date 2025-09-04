from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from hsi.data.aerorit import load_aerorit
from hsi.viz import make_rgb_from_cube
from hsi.data.patches import (
    create_patches, label_patches, remove_unlabeled_patches,
    coords_from_grid, expand_with_overlap_from_grid
)

# ---------- settings ----------
ROOT = Path(r"C:\Users\Nikolai A\Documents\Thesis\data\aerorit")
PATCH_SIZE     = 65
OVERLAP_STRIDE = 9         # must be < PATCH_SIZE
TRAIN_P, VAL_P = 0.33, 0.33
SEED           = 42
IGNORE_ID      = 3         # your "unlabeled" class id
# ------------------------------

# --- load & RGB ---
refl, labels, palette = load_aerorit(ROOT)
rgb = make_rgb_from_cube(refl, (650, 550, 450))
H, W = rgb.shape[:2]

# --- step 1: non-overlap split grid ---
grid = create_patches(refl, PATCH_SIZE)
split_grid = label_patches(grid, TRAIN_P, VAL_P, SEED)
split_grid = remove_unlabeled_patches(split_grid, labels, IGNORE_ID, PATCH_SIZE)

# --- base (non-overlap) coords ---
train_base = coords_from_grid(split_grid, PATCH_SIZE, 1)
val_base   = coords_from_grid(split_grid, PATCH_SIZE, 2)
test_base  = coords_from_grid(split_grid, PATCH_SIZE, 3)

# --- expand train/val with overlaps (test unchanged) ---
train_all = expand_with_overlap_from_grid(split_grid, 1, PATCH_SIZE, OVERLAP_STRIDE, image_shape=(H, W))
val_all   = expand_with_overlap_from_grid(split_grid, 2, PATCH_SIZE, OVERLAP_STRIDE, image_shape=(H, W))
test_all  = coords_from_grid(split_grid, PATCH_SIZE, 3) 

#test_all = test_base

# --- compute ONLY the extra (overlap) coords so we can outline them ---
def only_extra(all_coords: np.ndarray, base_coords: np.ndarray) -> np.ndarray:
    if all_coords.size == 0:
        return all_coords
    if base_coords.size == 0:
        return all_coords
    dt = np.dtype([('r', np.int32), ('c', np.int32)])
    a = all_coords.view(dt)
    b = base_coords.view(dt)
    extra = np.setdiff1d(a, b, assume_unique=False).view(np.int32).reshape(-1, 2)
    return extra

train_extra = only_extra(train_all, train_base)
val_extra   = only_extra(val_all,   val_base)

print(f"train base={len(train_base)}  -> all={len(train_all)}  (+{len(train_extra)})")
print(f"val   base={len(val_base)}    -> all={len(val_all)}    (+{len(val_extra)})")
print(f"test  base={len(test_base)}")

# --- plot: fill base patches, outline overlap patches ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(rgb)
ax.set_axis_off()
ax.set_title(f"P={PATCH_SIZE}, stride={OVERLAP_STRIDE} | "
             f"train {len(train_all)} (base {len(train_base)}), "
             f"val {len(val_all)} (base {len(val_base)}), "
             f"test {len(test_all)})")

def fill_rects(ax, coords, color, alpha, P):
    if coords.size == 0:
        return
    for r, c in coords:
        ax.add_patch(patches.Rectangle((int(c), int(r)), P, P,
                                       linewidth=0, facecolor=(*color, alpha), edgecolor=None))

def outline_rects(ax, coords, color, lw, P):
    if coords.size == 0:
        return
    for r, c in coords:
        ax.add_patch(patches.Rectangle((int(c), int(r)), P, P,
                                       linewidth=lw, edgecolor=color, facecolor='none'))

# colors (RGB 0..1)
c_train = (0/255, 180/255, 80/255)
c_val   = (255/255, 165/255, 0/255)
c_test  = (70/255, 130/255, 180/255)

# Fill base patches lightly
fill_rects(ax, train_base, c_train, alpha=0.22, P=PATCH_SIZE)
fill_rects(ax, val_base,   c_val,   alpha=0.22, P=PATCH_SIZE)
fill_rects(ax, test_base,  c_test,  alpha=0.22, P=PATCH_SIZE)

# Outline overlap patches boldly
outline_rects(ax, train_extra, c_train, lw=0.8, P=PATCH_SIZE)
outline_rects(ax, val_extra,   c_val,   lw=0.8, P=PATCH_SIZE)

plt.tight_layout()
plt.show()
