"""Contrast enhancement, field-of-view detection, and image-quality scoring."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.exposure import equalize_adapthist
from skimage.filters import laplace


def enhance(img: np.ndarray, clahe: bool = True, clahe_clip: float = 0.01, denoise: bool = True) -> np.ndarray:
    out = img.astype(np.float32)
    if clahe:
        out = equalize_adapthist(out, clip_limit=clahe_clip).astype(np.float32)
    if denoise:
        out = ndi.median_filter(out, size=3)
    lo, hi = np.percentile(out, [1, 99])
    return np.clip((out - lo) / max(float(hi - lo), 1e-6), 0, 1)


def field_of_view(img: np.ndarray) -> np.ndarray:
    """Mask of the collimated imaging field; excludes black borders and letterboxing."""
    from skimage.filters import threshold_otsu
    from skimage.morphology import disk

    blur = ndi.gaussian_filter(img, 8)
    thr = max(0.05, 0.5 * float(threshold_otsu(img)))
    fov = ndi.binary_closing(blur > thr, structure=disk(10))
    lbl, n = ndi.label(fov)
    if n > 1:
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        fov = lbl == int(np.argmax(sizes))
    fov = ndi.binary_fill_holes(fov)
    if fov.mean() < 0.02:
        fov = np.ones_like(fov, dtype=bool)
    return fov


def quality(img: np.ndarray, fov: np.ndarray) -> dict:
    vals = img[fov]
    contrast = float(vals.std())
    sharpness = float(laplace(ndi.gaussian_filter(img, 1.0))[fov].var())
    if contrast >= 0.15 and sharpness >= 1.5e-4:
        label = "adequate"
    elif contrast >= 0.08:
        label = "limited"
    else:
        label = "poor"
    return {"contrast": round(contrast, 4), "sharpness": round(sharpness, 6), "label": label}
