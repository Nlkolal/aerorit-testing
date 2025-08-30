# src/hsi/transforms/sample_coords.py
from __future__ import annotations
import numpy as np
from typing import Iterable, Tuple, Sequence

def id_from_rgb(palette: np.ndarray, rgb: Sequence[int]) -> int:
    """Return the integer ID in `palette` that matches an (R,G,B)."""
    pal = np.asarray(palette, dtype=np.uint8)
    target = np.array(rgb, dtype=np.uint8)
    idx = np.where((pal == target).all(axis=1))[0]
    if idx.size == 0:
        raise ValueError(f"RGB {tuple(target.tolist())} not found in palette")
    return int(idx[0])

def sample_coords_by_label(
    labels: np.ndarray,
    target_id: int,
    n: int = 50,
    seed: int = 42,
    min_spacing: int = 0,   # pixels; 0 = no spacing constraint
) -> list[Tuple[int,int]]:
    """
    Return up to N (row,col) coords where labels == target_id.
    If min_spacing > 0, uses a simple greedy selection to spread points.
    """
    rows, cols = np.where(labels == target_id)
    if rows.size == 0:
        return []

    rng = np.random.default_rng(seed)
    order = rng.permutation(rows.size)
    rows, cols = rows[order], cols[order]

    if min_spacing <= 0:
        k = min(n, rows.size)
        return [(int(rows[i]), int(cols[i])) for i in range(k)]

    # Greedy Poisson-disk-ish selection
    picked: list[Tuple[int,int]] = []
    for r, c in zip(rows, cols):
        if not picked:
            picked.append((int(r), int(c)))
            if len(picked) >= n:
                break
            continue
        rr = np.array([pr for pr, _ in picked])
        cc = np.array([pc for _, pc in picked])
        if np.all((np.abs(rr - r) >= min_spacing) | (np.abs(cc - c) >= min_spacing)):
            picked.append((int(r), int(c)))
            if len(picked) >= n:
                break
    return picked
