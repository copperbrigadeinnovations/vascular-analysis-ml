"""Vessel enhancement and binary segmentation (Frangi vesselness + hysteresis threshold)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold, frangi, threshold_otsu
from skimage.morphology import disk, skeletonize

from .sanity import graphic_artifacts


def default_sigmas(shape: tuple[int, int]) -> np.ndarray:
    """Scales cover vessel radii ~1px to ~7px; beyond ~10px the Hessian starts
    responding to background intensity structure rather than vessel ridges."""
    smax = float(np.clip(min(shape) / 40.0, 6.0, 10.0))
    return np.arange(1.0, smax + 0.5, 1.0)


def vesselness(img: np.ndarray, sigmas: np.ndarray | None = None) -> np.ndarray:
    sigmas = sigmas if sigmas is not None else default_sigmas(img.shape)
    # skimage's frangi mutates its `gamma` argument inside the scale loop, so
    # passing several sigmas at once locks gamma to the first scale and
    # suppresses large vessels; run each scale separately and take the max,
    # which is what a multi-scale Frangi filter is defined to return
    acc = np.zeros(img.shape, np.float32)
    for s in sigmas:
        v = frangi(img, sigmas=[float(s)], black_ridges=False).astype(np.float32)
        np.maximum(acc, v, out=acc)
    return acc / max(float(acc.max()), 1e-8)


def _remove_small_objects(mask: np.ndarray, min_px: int) -> np.ndarray:
    lbl, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lbl.ravel())
    keep = np.where(sizes >= min_px)[0]
    keep = keep[keep != 0]
    return np.isin(lbl, keep)


def _fill_small_holes(mask: np.ndarray, min_hole_px: int) -> np.ndarray:
    filled = ndi.binary_fill_holes(mask)
    holes = filled & ~mask
    lbl, n = ndi.label(holes)
    if n == 0:
        return mask
    sizes = np.bincount(lbl.ravel(), minlength=n + 1)
    small_ids = [i for i in range(1, n + 1) if sizes[i] < min_hole_px]
    if not small_ids:
        return mask
    return mask | np.isin(lbl, small_ids)


def segment(
    img: np.ndarray,
    fov: np.ndarray,
    sigmas: np.ndarray | None = None,
    min_size_px: int = 64,
    min_hole_px: int = 64,
    raw: np.ndarray | None = None,
) -> dict:
    v = vesselness(img, sigmas)
    high = max(float(threshold_otsu(v)), 0.05)
    mask = apply_hysteresis_threshold(v, 0.5 * high, high)
    mask &= fov
    mask = ndi.binary_closing(mask, structure=disk(2))
    mask = _remove_small_objects(mask, min_size_px)
    mask = _fill_small_holes(mask, min_hole_px)

    # strip flat-intensity, saturated bars (panel dividers, scale bars, annotation
    # rules) before centerline extraction: they are never vessels, but a thin bright
    # bar can otherwise skeletonize into a long "branch" with a false candidate
    # stenosis wherever its edge is slightly irregular. Detected on `raw` (the
    # pre-adaptive-contrast image) since CLAHE amplifies a bar's own compression
    # noise into something that no longer reads as flat.
    artifacts, artifact_mask = graphic_artifacts(raw if raw is not None else img, fov)
    if artifact_mask.any():
        mask = mask & ~artifact_mask

    skel = skeletonize(mask)
    return {"mask": mask, "skeleton": skel, "vesselness": v, "threshold": high, "artifacts": artifacts}
