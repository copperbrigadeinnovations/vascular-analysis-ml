"""Heuristic pre-flight checks for non-vascular artefacts and composite-figure layouts
that can corrupt automated vessel segmentation and quantification.

These are deterministic, rule-based image checks (no classifier/model involved) and are
deliberately conservative: most flags mean "a human should look at this", not a verdict
on the image. The one exception is `graphic_artifacts`, which identifies image furniture
(panel dividers, scale bars, annotation rules) with high enough confidence that it is
also stripped from the vessel mask before quantification, since a flat-intensity,
perfectly straight rectangle is never a vessel.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops


def _label(binary: np.ndarray) -> tuple[np.ndarray, int]:
    return ndi.label(binary, structure=np.ones((3, 3), bool))


def panel_layout(img: np.ndarray, bright_thresh: float = 0.15, min_gap_frac: float = 0.02, min_side_frac: float = 0.1) -> dict:
    """Flag an image that splits into independent left/right panels -- a vertical
    corridor with no image content at all, bounded by substantial content on both
    sides. This looks directly at raw brightness rather than the (heavily blurred and
    closed) field-of-view mask used for segmentation: that mask is tuned to bridge
    small gaps *within* one sparse angiogram, which is exactly wrong for telling two
    genuinely separate panels apart -- it would merge them too.
    """
    h, w = img.shape
    col_has_content = (img > bright_thresh).any(axis=0)
    gaps, start = [], None
    for i, v in enumerate(col_has_content):
        if not v and start is None:
            start = i
        elif v and start is not None:
            gaps.append((start, i))
            start = None
    if start is not None:
        gaps.append((start, len(col_has_content)))

    # interior gaps only: not the empty margin before the first / after the last
    # piece of content, and wide enough to be a deliberate divider
    interior = [(a, b) for a, b in gaps if a > 0 and b < w and (b - a) >= min_gap_frac * w]
    for a, b in interior:
        left_frac = float(col_has_content[:a].sum()) / w
        right_frac = float(col_has_content[b:].sum()) / w
        if left_frac >= min_side_frac and right_frac >= min_side_frac:
            return {"multi_panel": True, "gap": [int(a), int(b)]}
    return {"multi_panel": False, "gap": None}


def _bar_bands(flagged: np.ndarray) -> list[tuple[int, int]]:
    """Group a boolean 1D array into (start, stop) index ranges of consecutive True runs."""
    bands = []
    start = None
    for i, v in enumerate(flagged):
        if v and start is None:
            start = i
        elif not v and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, len(flagged)))
    return bands


def graphic_artifacts(
    raw_img: np.ndarray,
    fov: np.ndarray,
    min_len_frac: float = 0.5,
    max_thickness_frac: float = 0.06,
    flat_std: float = 0.02,
    extreme_lo: float = 0.08,
    extreme_hi: float = 0.92,
    end_pad_px: int = 6,
) -> tuple[list[dict], np.ndarray]:
    """Identify long, flat, saturated (near-pure white or near-pure black) bars -- panel
    dividers, scale bars, annotation rules -- directly from pixel intensities on the
    *pre-enhancement* image, before adaptive contrast equalisation. Adaptive histogram
    equalisation stretches local contrast everywhere, including a bar's own compression
    noise, which would otherwise mask the one property that gives it away: a real vessel
    tapers and carries texture along its length, a synthetic divider does not.

    Returns the flagged rectangles plus a boolean mask of the pixels responsible, so
    callers can strip them out of the vessel mask before quantification.
    """
    h, w = raw_img.shape
    found: list[dict] = []
    artifact_mask = np.zeros_like(fov, dtype=bool)

    # a divider bar is often drawn full-bleed, taller than the detected anatomical
    # field-of-view (collimator) region -- so flatness/extremity is measured only over
    # each column/row's in-FOV pixels, not pulled down by substituting a background
    # median for the small margin outside the FOV (which would falsely look "noisy")
    masked = np.where(fov, raw_img, np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_std, col_mean = np.nanstd(masked, axis=0), np.nanmean(masked, axis=0)
        row_std, row_mean = np.nanstd(masked, axis=1), np.nanmean(masked, axis=1)
    col_std, col_mean = np.nan_to_num(col_std, nan=1.0), np.nan_to_num(col_mean, nan=0.5)
    row_std, row_mean = np.nan_to_num(row_std, nan=1.0), np.nan_to_num(row_mean, nan=0.5)

    def _extreme_flat(values_std, values_mean, coverage):
        return (values_std <= flat_std) & ((values_mean >= extreme_hi) | (values_mean <= extreme_lo)) & (coverage >= 0.5)

    # vertical bars: scan columns. Matching tolerates a minority of off-value columns
    # within the band (anti-aliased edge columns at the bar's boundary) by requiring
    # most, not all, columns to agree with the dominant value on a given row.
    col_flat = _extreme_flat(col_std, col_mean, fov.mean(axis=0))
    for c0, c1 in _bar_bands(col_flat):
        if c1 - c0 > max_thickness_frac * w:
            continue
        band, band_fov = raw_img[:, c0:c1], fov[:, c0:c1].all(axis=1)
        if not band_fov.any():
            continue
        # the dominant value is estimated from in-FOV pixels only (robust to the
        # fov-excluded margin), but matched against every row regardless of FOV: a
        # divider bar is commonly drawn full-bleed, extending past the detected
        # anatomical field, and those extending pixels are still real bar pixels
        dominant = float(np.median(band[band_fov]))
        close_frac = np.mean(np.abs(band - dominant) <= 0.05, axis=1)
        rows = np.where(close_frac >= 0.7)[0]
        if len(rows) == 0 or (rows.max() - rows.min() + 1) < min_len_frac * h:
            continue
        r0, r1 = max(0, int(rows.min()) - end_pad_px), min(h, int(rows.max()) + 1 + end_pad_px)
        bbox = [r0, c0, r1, c1]
        artifact_mask[bbox[0] : bbox[2], bbox[1] : bbox[3]] = True
        found.append({"bbox": bbox, "orientation": "vertical", "length_px": int(rows.max() - rows.min() + 1)})

    # horizontal bars: scan rows (same tolerance logic, transposed)
    row_flat = _extreme_flat(row_std, row_mean, fov.mean(axis=1))
    for r0, r1 in _bar_bands(row_flat):
        if r1 - r0 > max_thickness_frac * h:
            continue
        band, band_fov = raw_img[r0:r1, :], fov[r0:r1, :].all(axis=0)
        if not band_fov.any():
            continue
        dominant = float(np.median(band[:, band_fov]))
        close_frac = np.mean(np.abs(band - dominant) <= 0.05, axis=0)
        cols = np.where(close_frac >= 0.7)[0]
        if len(cols) == 0 or (cols.max() - cols.min() + 1) < min_len_frac * w:
            continue
        c0, c1 = max(0, int(cols.min()) - end_pad_px), min(w, int(cols.max()) + 1 + end_pad_px)
        bbox = [r0, c0, r1, c1]
        artifact_mask[bbox[0] : bbox[2], bbox[1] : bbox[3]] = True
        found.append({"bbox": bbox, "orientation": "horizontal", "length_px": int(cols.max() - cols.min() + 1)})

    found.sort(key=lambda d: -d["length_px"])
    return found, artifact_mask


def seam_lines(img: np.ndarray, fov: np.ndarray, z_thresh: float = 6.0, min_coverage: float = 0.6) -> dict:
    """Detect near-complete rows with an abnormally sharp, uniform intensity jump versus
    their neighbours: the signature of a station-junction / stitch line in a multi-station
    composite, rather than an anatomical edge (which is local, not full-width)."""
    coverage = fov.mean(axis=1)
    rows = np.where(coverage >= min_coverage)[0]
    if len(rows) < 5:
        return {"n_seams": 0, "rows": []}
    diffs = np.abs(np.diff(img[rows, :], axis=0)).mean(axis=1)
    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med))) + 1e-6
    z = (diffs - med) / (1.4826 * mad)
    flagged = rows[1:][z > z_thresh]
    groups: list[list[int]] = []
    for r in flagged:
        if groups and r - groups[-1][-1] <= 2:
            groups[-1].append(int(r))
        else:
            groups.append([int(r)])
    return {"n_seams": len(groups), "rows": [int(np.mean(g)) for g in groups]}


def label_like_blobs(
    img: np.ndarray,
    fov: np.ndarray,
    mask: np.ndarray,
    min_area: int = 8,
    max_area_frac: float = 0.002,
) -> dict:
    """Flag small, solid, blocky bright regions outside the vessel mask: typically panel
    letters ('a'/'b'), scale numbers, or orientation markers burned into the image, which
    can otherwise be mistaken for short spurious vessel fragments."""
    from skimage.filters import threshold_otsu

    bright = (img > max(0.85, float(threshold_otsu(img[fov])))) & fov & ~ndi.binary_dilation(mask, iterations=2)
    lbl, n = _label(bright)
    if n == 0:
        return {"n_candidates": 0, "locations": []}
    max_area = max_area_frac * float(fov.sum())
    found = []
    for rp in regionprops(lbl):
        if rp.area < min_area or rp.area > max_area:
            continue
        h = rp.bbox[2] - rp.bbox[0]
        w = rp.bbox[3] - rp.bbox[1]
        extent = rp.area / max(h * w, 1)
        elong = max(h, w) / max(min(h, w), 1)
        if extent >= 0.4 and elong <= 3.5 and rp.solidity >= 0.55:
            cy, cx = rp.centroid
            found.append([int(cy), int(cx)])
    found = found[:10]
    return {"n_candidates": len(found), "locations": found}
