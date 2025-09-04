# src/hsi/data/patches.py
import numpy as np
from typing import Iterable

def create_patches(hyper_data: np.ndarray, patch_size: int):
    im_height = hyper_data.shape[0]
    im_width  = hyper_data.shape[1]
    height_divs = im_height // patch_size
    width_divs  = im_width  // patch_size
    patch_array = np.zeros((height_divs, width_divs), dtype=np.int32)
    return patch_array

def label_patches(patch_array: np.ndarray, train_percent: float, validate_percent: float, SEED):
    assert 0.0 <= train_percent <= 1.0
    assert 0.0 <= validate_percent <= 1.0
    assert train_percent + validate_percent <= 1.0

    H, W = patch_array.shape
    total_patches = H * W
    n_train    = int(round(total_patches * train_percent))
    n_validate = int(round(total_patches * validate_percent))
    n_test     = total_patches - n_train - n_validate

    rng = np.random.default_rng(SEED)
    order = rng.permutation(total_patches)

    flat = np.zeros(total_patches, dtype=np.int32)
    flat[order[:n_train]] = 1
    flat[order[n_train:n_train + n_validate]] = 2
    flat[order[n_train + n_validate:]] = 3
    return flat.reshape(H, W)

def coords_from_grid(grid: np.ndarray, patch_size: int, values) -> np.ndarray:
    """Top-left coords for any grid cell whose value is in `values`."""
    if np.isscalar(values):
        values = [int(values)]
    ys, xs = np.where(np.isin(grid, list(values)))
    return np.stack([ys * patch_size, xs * patch_size], axis=1).astype(np.int32)

def remove_unlabeled_patches(patch_array: np.ndarray, labels: np.ndarray, label_id_discard: int, patch_size: int) -> np.ndarray:
    """Mark a patch as -1 if it contains *any* pixel with label_id_discard."""
    hdiv, wdiv = patch_array.shape
    ph = pw = int(patch_size)
    out = patch_array.copy()
    for i in range(hdiv):
        y0, y1 = i * ph, i * ph + ph
        for j in range(wdiv):
            x0, x1 = j * pw, j * pw + pw
            if y1 > labels.shape[0] or x1 > labels.shape[1]:
                continue
            if (labels[y0:y1, x0:x1] == label_id_discard).any():
                out[i, j] = -1
    return out

def make_dense_index(H: int, W: int, patch: int, stride: int) -> np.ndarray:
    rows = np.arange(0, H - patch + 1, stride, dtype=np.int32)
    cols = np.arange(0, W - patch + 1, stride, dtype=np.int32)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return np.stack([rr.ravel(), cc.ravel()], axis=1)

def contained(mask: np.ndarray, r: int, c: int, patch: int) -> bool:
    return mask[r:r+patch, c:c+patch].all()

def overlap_coverage_from_grid(split_grid: np.ndarray, split_value: int, patch: int) -> np.ndarray:
    """Coverage where there is at least one horizontal or vertical adjacency of base patches."""
    hdiv, wdiv = split_grid.shape
    Hc, Wc = hdiv * patch, wdiv * patch
    cov = np.zeros((Hc, Wc), dtype=bool)

    g = (split_grid == split_value)

    if wdiv > 1:
        pair_h = g[:, :-1] & g[:, 1:]
        ih, jh = np.where(pair_h)
        for i, j in zip(ih, jh):
            y0, y1 = i * patch, (i + 1) * patch
            x0, x1 = j * patch, (j + 2) * patch
            cov[y0:y1, x0:x1] = True

    if hdiv > 1:
        pair_v = g[:-1, :] & g[1:, :]
        iv, jv = np.where(pair_v)
        for i, j in zip(iv, jv):
            y0, y1 = i * patch, (i + 2) * patch
            x0, x1 = j * patch, (j + 1) * patch
            cov[y0:y1, x0:x1] = True
    return cov

import numpy as np

def expand_with_overlap_from_grid(
    split_grid: np.ndarray,
    split_value: int,
    patch: int,
    stride: int,
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Overlap *anchored to each adjacency strip*, so starts go from the strip edge in steps of `stride`
    and we force-include the far edge (x0+P / y0+P). Eliminates the 'tip gap'.
    """
    hdiv, wdiv = split_grid.shape
    Hc, Wc = hdiv * patch, wdiv * patch
    if image_shape is None:
        H, W = Hc, Wc
    else:
        H, W = min(image_shape[0], Hc), min(image_shape[1], Wc)

    # base (non-overlap) coords
    base = coords_from_grid(split_grid, patch, split_value)
    coords = [tuple(rc) for rc in base.tolist()]

    g = (split_grid == split_value)

    # --- horizontal adjacency strips (one patch tall, two patches wide) ---
    if wdiv > 1:
        pair_h = g[:, :-1] & g[:, 1:]
        ih, jh = np.where(pair_h)
        for i, j in zip(ih, jh):
            y0 = i * patch
            x_left = j * patch
            # starts anchored at the strip's left edge; include far edge x_left+patch
            x_starts = list(range(x_left, x_left + patch + 1, stride))
            if x_starts[-1] != x_left + patch:
                x_starts.append(x_left + patch)
            for x in x_starts:
                if x + patch <= W:
                    coords.append((y0, x))

    # --- vertical adjacency strips (two patches tall, one patch wide) ---
    if hdiv > 1:
        pair_v = g[:-1, :] & g[1:, :]
        iv, jv = np.where(pair_v)
        for i, j in zip(iv, jv):
            y_top = i * patch
            x0 = j * patch
            # starts anchored at the strip's top edge; include far edge y_top+patch
            y_starts = list(range(y_top, y_top + patch + 1, stride))
            if y_starts[-1] != y_top + patch:
                y_starts.append(y_top + patch)
            for y in y_starts:
                if y + patch <= H:
                    coords.append((y, x0))

    allc = np.unique(np.array(coords, dtype=np.int32).reshape(-1, 2), axis=0)
    return allc












