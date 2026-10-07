"""Centerline branch decomposition, calibre profiling, tortuosity, and candidate stenosis detection.

Values are returned in pixels unless spacing_mm is supplied, in which case lengths and
diameters are in millimetres. Physical-unit outputs assume the 2D projection is not
foreshortened and that spacing is isotropic.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

_NEIGH_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


def _neighbor_count(skel: np.ndarray) -> np.ndarray:
    return ndi.convolve(skel.astype(np.uint8), _NEIGH_KERNEL, mode="constant")


def _order_path(coords: np.ndarray) -> np.ndarray:
    """Order an 8-connected simple path from one end to the other."""
    pts = {tuple(c) for c in coords}

    def nbrs(p):
        y, x = p
        return [
            q
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dy or dx) and (q := (y + dy, x + dx)) in pts
        ]

    ends = [p for p in pts if len(nbrs(p)) == 1]
    if not ends:
        ends = [next(iter(pts))]
    path, seen, cur = [], set(), ends[0]
    while cur is not None:
        path.append(cur)
        seen.add(cur)
        nxt = [q for q in nbrs(cur) if q not in seen]
        cur = nxt[0] if nxt else None
    return np.asarray(path, dtype=float)


def _arc_chord(path: np.ndarray, sy: float, sx: float) -> tuple[float, float]:
    if len(path) < 2:
        return 0.0, 0.0
    d = np.diff(path, axis=0)
    arc = float(np.hypot(d[:, 0] * sy, d[:, 1] * sx).sum())
    chord = float(np.hypot((path[-1, 0] - path[0, 0]) * sy, (path[-1, 1] - path[0, 1]) * sx))
    return arc, chord


def _longest_run(flags: np.ndarray) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def _analyse_branch(
    path: np.ndarray,
    dist: np.ndarray,
    sy: float,
    sx: float,
    stenosis_ratio: float,
) -> dict:
    arc, chord = _arc_chord(path, sy, sx)
    rows = path[:, 0].astype(int)
    cols = path[:, 1].astype(int)
    scale = 0.5 * (sy + sx)
    diam = 2.0 * dist[rows, cols] * scale

    prof = diam.astype(float)
    k = min(5, len(prof) if len(prof) % 2 == 1 else len(prof) - 1)
    if k >= 3:
        prof = ndi.median_filter(prof, size=k)

    n = len(prof)
    # branch ends are cut by junction removal or taper into a free termination,
    # where the distance transform under-reports calibre; exclude those margins
    margin = min(12, max(4, n // 8))
    core = np.arange(margin, n - margin) if n > 2 * margin else np.arange(n)

    # local reference diameter (rolling 90th percentile) so normal tapering does
    # not read as stenosis, only genuinely focal dips relative to neighbours do
    w = min(31, n if n % 2 == 1 else n - 1)
    if w >= 5:
        ref_prof = ndi.percentile_filter(prof, percentile=90, size=w, mode="nearest")
    else:
        ref_prof = np.full_like(prof, np.percentile(prof, 90))

    ratio_prof = prof / np.maximum(ref_prof, 1e-6)
    i_min = int(core[int(np.argmin(ratio_prof[core]))])
    narrow = ratio_prof[core] < stenosis_ratio
    narrowing_run = _longest_run(narrow)
    # a real stenosis is a dip founded on both sides: the ratio recovers above the
    # threshold inside the core. If the narrowing run containing the minimum reaches
    # a core boundary it is an ostial dip or a taper into a free termination instead
    i_core = int(np.argmin(ratio_prof[core]))
    lo = hi = i_core
    while lo > 0 and narrow[lo - 1]:
        lo -= 1
    while hi < len(narrow) - 1 and narrow[hi + 1]:
        hi += 1
    truncated = bool(len(narrow) and narrow[i_core] and (lo == 0 or hi == len(narrow) - 1))

    return {
        "length": round(arc, 3),
        "chord": round(chord, 3),
        "tortuosity": round(arc / chord, 3) if chord > 1e-6 else None,
        "mean_diam": round(float(prof.mean()), 3),
        "min_diam": round(float(prof[i_min]), 3),
        "ref_diam": round(float(ref_prof[i_min]), 3),
        "min_ratio": round(float(ratio_prof[i_min]), 3),
        "narrowing_run": int(narrowing_run),
        "narrowing_truncated": truncated,
        "min_pos_frac": round(i_min / max(n - 1, 1), 3),
        "min_rc": [int(rows[i_min]), int(cols[i_min])],
        "n_points": int(n),
    }


def analyze(
    mask: np.ndarray,
    skeleton: np.ndarray,
    spacing_mm: tuple[float, float] | None = None,
    min_branch_px: int = 12,
    stenosis_ratio: float = 0.7,
    stenosis_min_points: int = 25,
    stenosis_min_run: int = 6,
    min_ref_diam_px: float = 5.0,
) -> dict:
    sy, sx = spacing_mm if spacing_mm else (1.0, 1.0)
    scale = 0.5 * (sy + sx)
    # below ~5-6px reference calibre, pixel quantisation and anti-aliasing noise are not
    # reliably distinguishable from a true focal narrowing, so candidates this thin are
    # tracked separately rather than reported as findings
    min_ref_diam = min_ref_diam_px * scale
    dist = ndi.distance_transform_edt(mask)

    ncount = _neighbor_count(skeleton)
    junctions = skeleton & (ncount >= 3)
    endpoints = skeleton & (ncount == 1)
    _, n_bifurcations = ndi.label(junctions, structure=np.ones((3, 3), bool))

    pruned = skeleton & ~ndi.binary_dilation(junctions, np.ones((3, 3), bool))
    branches_lbl, n_branches_raw = ndi.label(pruned, structure=np.ones((3, 3), bool))

    branches = []
    for i in range(1, n_branches_raw + 1):
        coords = np.argwhere(branches_lbl == i)
        if len(coords) < min_branch_px:
            continue
        path = _order_path(coords)
        branch = _analyse_branch(path, dist, sy, sx, stenosis_ratio)
        if branch["length"] <= 0:
            continue
        branches.append(branch)

    lengths = np.array([b["length"] for b in branches])
    diams = np.array([b["mean_diam"] for b in branches])
    torts = np.array([b["tortuosity"] for b in branches if b["tortuosity"] is not None])

    qualified = [
        b
        for b in branches
        if b["min_ratio"] < stenosis_ratio
        # sub-6-px runs are transient calibre-step dips where the rolling
        # reference lags at a taper; a real focal lesion has longitudinal extent
        and b["narrowing_run"] >= stenosis_min_run
        and not b["narrowing_truncated"]
        and b["n_points"] >= stenosis_min_points
    ]
    candidates = sorted((b for b in qualified if b["ref_diam"] >= min_ref_diam), key=lambda b: b["min_ratio"])
    n_suppressed_small_caliber = sum(1 for b in qualified if b["ref_diam"] < min_ref_diam)

    by_length = sorted(branches, key=lambda b: -b["length"])
    return {
        "n_branches": int(len(branches)),
        "n_endpoints": int(endpoints.sum()),
        "n_bifurcations": int(n_bifurcations),
        "total_length": round(float(lengths.sum()), 3) if len(lengths) else 0.0,
        "mean_branch_diam": round(float(np.average(diams, weights=lengths)), 3) if len(diams) else 0.0,
        "max_branch_diam": round(float(diams.max()), 3) if len(diams) else 0.0,
        "min_branch_diam": round(float(diams.min()), 3) if len(diams) else 0.0,
        "mean_tortuosity": round(float(np.median(torts)), 3) if len(torts) else None,
        "max_tortuosity": round(float(torts.max()), 3) if len(torts) else None,
        "stenosis_ratio_threshold": stenosis_ratio,
        "candidate_stenoses": candidates,
        "min_ref_diam": round(float(min_ref_diam), 3),
        "n_suppressed_small_caliber": int(n_suppressed_small_caliber),
        "branches": by_length[:20],
        "spacing_mm": [sy, sx] if spacing_mm else None,
    }
