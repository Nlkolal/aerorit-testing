# src/hsi/viz/__init__.py
from .rgb import (
    make_rgb_from_cube,
    show_img,
    show_band,
    percentile_stretch,
    wavelength_to_index,
)

from .spectra import (
    plot_spectrum,
    plot_spectra,
    spectra_heatmap,
    nm_axis,
    nm_to_thz
)

__all__ = [
    # rgb
    "make_rgb_from_cube",
    "show_img",
    "show_band",
    "percentile_stretch",
    "wavelength_to_index",
    # spectra
    "plot_spectrum",
    "plot_spectra",
    "spectra_heatmap",
    "nm_axis",
    "nm_to_thz",
]
