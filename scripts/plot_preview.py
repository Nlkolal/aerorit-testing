from pathlib import Path
from hsi.data.aerorit import load_aerorit
from hsi.data.common import load_tif  # only for rgb/radiance

ROOT = Path(r"C:\Users\Nikolai A\Documents\Thesis\data\aerorit")

refl, labels, palette = load_aerorit(ROOT, save_palette=True)  # reflectance + labels
rgb = load_tif(ROOT / "image_rgb.tif")
rad = load_tif(ROOT / "image_hsi_radiance.tif")

print("Reflectance:", refl.shape, refl.dtype)
print("Radiance   :", rad.shape,  rad.dtype)
print("RGB        :", rgb.shape,  rgb.dtype)
print("Labels     :", labels.shape, labels.dtype)
print("Palette    :", None if palette is None or palette.size == 0 else palette.shape)