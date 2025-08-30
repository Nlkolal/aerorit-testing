from pathlib import Path
import numpy as np
from hsi.data.aerorit import load_aerorit
from hsi.viz import plot_spectrum, plot_spectra, spectra_heatmap

# <<< EDIT THESE >>>
ROOT     = Path(r"C:\Users\Nikolai A\Documents\Thesis\data\aerorit")
MODE     = "nm"                 # or "thz"
OUT      = None                 # e.g., Path("plots/spectra.png") to save
HEATMAP  = True

# Option A: give explicit coords (comment this out to use label-based sampling)
COORDS   = None         # one or many (row, col)



# Option B: build coords by LABEL (uncomment ONE of these)
SELECT_BY_ID   = 5           # e.g., 3
SELECT_BY_RGB  = None           # e.g., (0, 0, 255)  # palette color for the class
N_SAMPLES      = 1000
SEED           = 42

Y_MIN, Y_MAX = 0.0, 0.006

def _id_from_rgb(palette: np.ndarray, rgb_triplet):
    rgb = np.array(rgb_triplet, dtype=np.uint8)
    idx = np.where((palette == rgb).all(axis=1))[0]
    if idx.size == 0:
        raise ValueError(f"RGB {tuple(rgb.tolist())} not found in palette")
    return int(idx[0])

def _sample_coords_by_label(labels: np.ndarray, target_id: int, n=50, seed=42):
    rows, cols = np.where(labels == target_id)
    if rows.size == 0:
        return []
    rng = np.random.default_rng(seed)
    take = min(n, rows.size)
    sel = rng.choice(rows.size, size=take, replace=False)
    return [(int(rows[i]), int(cols[i])) for i in sel]

def run(root=ROOT, mode=MODE, out=OUT):
    # load_aerorit returns (reflectance, labels, palette)
    refl, labels, palette = load_aerorit(root)

    # Decide coords:
    coords = COORDS
    if not coords or len(coords) == 0:
        # build from label if requested
        if SELECT_BY_ID is not None or SELECT_BY_RGB is not None:
            if SELECT_BY_ID is None:
                target_id = _id_from_rgb(palette, SELECT_BY_RGB)
                print(f"[info] Using palette RGB {SELECT_BY_RGB} -> ID {target_id}")
            else:
                target_id = SELECT_BY_ID
                print(f"[info] Using ID {target_id}", " Pixel RGB:", palette[target_id].tolist())
            coords = _sample_coords_by_label(labels, target_id, n=N_SAMPLES, seed=SEED)
            if not coords:
                print("[warn] No pixels found for that label; falling back to image center.")
        if not coords:
            # default: image center
            H, W = labels.shape
            coords = [(H // 2, W // 2)]
            print(f"[info] Using center pixel {coords[0]}")

    # Plot
    if len(coords) == 1:
        r, c = coords[0]
        plot_spectrum(
            refl, r, c, mode=mode,
            title=f"Spectrum @ ({r},{c})",
            save_path=(str(out) if out else None),
        )
    else:
        plot_spectra(
            refl, coords, mode=mode, y_range=(Y_MIN, Y_MAX),
            title=f"{len(coords)} spectra cars",
            save_path=(str(out) if out else None),
        )
        if HEATMAP:
            # auto y-range so tiny values don't squash to the bottom
            save = None
            if out is not None:
                out = Path(out)
                save = out.with_stem(out.stem + "_heatmap")
            spectra_heatmap(
                refl, coords, mode=mode,
                reflect_range=(Y_MIN, Y_MAX),          
                bins_reflect=80,
                title="Aggregated spectral density cars",
                save_path=(str(save) if save else None),
            )

if __name__ == "__main__":
    run()
