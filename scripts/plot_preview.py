#cd "C:\Users\Nikolai A\Documents\Thesis"; git add -A; git commit -m "MESSAGE"; git pull --rebase origin main; git push origin main


from pathlib import Path
from hsi.data.aerorit import load_aerorit
from hsi.data.common import load_tif  # only for rgb/radiance
# if you re-exported in hsi/viz/__init__.py:
from hsi.viz import make_rgb_from_cube, show_img
# otherwise use: from hsi.viz.rgb import make_rgb_from_cube, show_img

ROOT = Path(r"C:\Users\Nikolai A\Documents\Thesis\data\aerorit")

# Load reflectance + labels (+palette if labels were RGB)
refl, labels, palette = load_aerorit(ROOT, save_palette=True)
# Also load the provided RGB and radiance (optional)
rgb = load_tif(ROOT / "image_rgb.tif")
rad = load_tif(ROOT / "image_hsi_radiance.tif")

print("Reflectance:", refl.shape, refl.dtype)
print("Radiance   :", rad.shape,  rad.dtype)
print("RGB        :", rgb.shape,  rgb.dtype)
print("Labels     :", labels.shape, labels.dtype)
print("Palette    :", None if palette is None or palette.size == 0 else palette.shape)

# ---- Plot RGB from the hyperspectral cube ----
# Natural-ish color (R,G,B ~ 650,550,450 nm)
rgb_from_cube = make_rgb_from_cube(refl, (650, 550, 450))
show_img(rgb_from_cube, "AeroRIT RGB (650/550/450 nm)", downsample=3)

# (Optional) False color: NIR/Red/Green (NIR highlights vegetation)
false_color = make_rgb_from_cube(refl, (800, 650, 550))
show_img(false_color, "AeroRIT False Color (800/650/550 nm)", downsample=3)

