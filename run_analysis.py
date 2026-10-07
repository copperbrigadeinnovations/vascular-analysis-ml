"""CLI: automatic vascular analysis of X-ray angiography images (research use only).

Usage:
    python run_analysis.py <file-or-folder> [--outdir outputs] [--spacing-mm 0.3]

Outputs per image: <name>_observation.txt (single paragraph), <name>_metrics.json,
<name>_overlay.png (segmentation + centerline + candidate narrowing markers).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours

from vascular.io_utils import load_image
from vascular.preprocess import enhance, field_of_view, quality
from vascular.quantify import analyze
from vascular.report import build_observation, build_recommendations
from vascular.sanity import label_like_blobs, panel_layout, seam_lines
from vascular.segment import segment

EXTS = {".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _overlay(path: Path, proc: np.ndarray, seg: dict, metrics: dict, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), facecolor="black")
    axes[0].imshow(proc, cmap="gray")
    axes[0].set_title("input (enhanced)", color="white")
    axes[1].imshow(proc, cmap="gray")
    for c in find_contours(seg["mask"], 0.5):
        axes[1].plot(c[:, 1], c[:, 0], color="orange", lw=0.8)
    axes[1].set_title("vessel segmentation", color="white")
    axes[2].imshow(proc, cmap="gray")
    sk = np.argwhere(seg["skeleton"])
    if len(sk):
        axes[2].scatter(sk[:, 1], sk[:, 0], s=0.3, c="red")
    for s in metrics["candidate_stenoses"][:5]:
        r, c = s["min_rc"]
        axes[2].scatter([c], [r], s=90, facecolors="none", edgecolors="cyan", lw=1.2)
    axes[2].set_title("centerline + candidate narrowing", color="white")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{path.name}  |  research output - not for diagnostic use", color="white", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor="black")
    plt.close(fig)


def analyze_file(
    path: Path,
    outdir: Path,
    spacing_mm: float | None = None,
    min_branch_px: int = 12,
    stenosis_ratio: float = 0.7,
    stenosis_min_run: int = 6,
    min_ref_diam_px: float = 5.0,
) -> tuple[str, dict, dict, str]:
    img, spacing, meta = load_image(path)
    if spacing_mm:
        spacing = (float(spacing_mm), float(spacing_mm))

    fov = field_of_view(img)
    # flatten the collimator/FOV step edge with the interior median before
    # CLAHE, otherwise large Frangi scales read that edge as a vessel ridge
    base = np.where(fov, img, np.median(img[fov]))
    proc = enhance(base)
    qual = quality(proc, fov)
    seg = segment(proc, fov, raw=img)

    density = float(np.logical_and(seg["mask"], fov).sum()) / max(int(fov.sum()), 1)
    metrics = analyze(
        seg["mask"],
        seg["skeleton"],
        spacing_mm=spacing,
        min_branch_px=min_branch_px,
        stenosis_ratio=stenosis_ratio,
        stenosis_min_run=stenosis_min_run,
        min_ref_diam_px=min_ref_diam_px,
    )
    unit = "mm" if spacing else "px"
    paragraph = build_observation(metrics, qual, meta, unit=unit, density=density, calibrated=bool(spacing))

    rows_idx, cols_idx = np.where(seg["mask"])
    vessel_aspect = None
    if len(rows_idx):
        vh = int(rows_idx.max() - rows_idx.min() + 1)
        vw = int(cols_idx.max() - cols_idx.min() + 1)
        vessel_aspect = vh / max(vw, 1)

    checks = {
        # raw img, not proc: CLAHE brightens a true empty gap enough to erase the
        # "no content here" signal a gap between separate panels depends on
        "panels": panel_layout(img),
        "artifacts": seg["artifacts"],
        "seams": seam_lines(proc, fov),
        "labels": label_like_blobs(proc, fov, seg["mask"]),
        "image_height": img.shape[0],
        "image_width": img.shape[1],
        "vessel_aspect": round(vessel_aspect, 3) if vessel_aspect else None,
    }
    recommendation = build_recommendations(checks, metrics, qual)
    full_text = paragraph + (f"\n\n{recommendation}" if recommendation else "")

    (outdir / f"{path.stem}_observation.txt").write_text(full_text, encoding="utf-8")
    with open(outdir / f"{path.stem}_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "file": str(path),
                "quality": qual,
                "units": unit,
                "metrics": metrics,
                "checks": checks,
                "observation": paragraph,
                "recommendation": recommendation,
            },
            fh,
            indent=2,
        )
    _overlay(path, proc, seg, metrics, outdir / f"{path.stem}_overlay.png")
    return paragraph, metrics, qual, recommendation


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Automatic vessel analysis of X-ray angiography (research use only).")
    ap.add_argument("input", help="image file or folder of images (DICOM / PNG / TIFF)")
    ap.add_argument("--outdir", default="outputs", help="output folder (default: ./outputs)")
    ap.add_argument("--spacing-mm", type=float, default=None, help="override pixel spacing (mm/px)")
    ap.add_argument("--min-branch-px", type=int, default=12, help="ignore centerline branches shorter than this")
    ap.add_argument("--stenosis-ratio", type=float, default=0.7, help="candidate narrowing threshold (default 0.7)")
    ap.add_argument("--min-lesion-px", type=int, default=6, help="minimum run of narrowed calibre (px) for a candidate (default 6)")
    ap.add_argument(
        "--min-ref-diam-px",
        type=float,
        default=5.0,
        help="ignore candidate narrowings whose reference calibre is below this many pixels (default 5.0)",
    )
    args = ap.parse_args(argv)

    in_path = Path(args.input)
    files = (
        [in_path]
        if in_path.is_file()
        else sorted(p for p in in_path.iterdir() if p.suffix.lower() in EXTS)
    )
    if not files:
        print(f"No supported images found at {in_path}")
        return 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for f in files:
        try:
            paragraph, _, qual, recommendation = analyze_file(
                f,
                outdir,
                args.spacing_mm,
                args.min_branch_px,
                args.stenosis_ratio,
                args.min_lesion_px,
                args.min_ref_diam_px,
            )
        except Exception as exc:
            print(f"[skip] {f.name}: {exc}")
            continue
        print(f"\n=== {f.name} (quality: {qual['label']}) ===")
        print(paragraph)
        if recommendation:
            print()
            print(recommendation)
        print(f"[saved] {outdir / (f.stem + '_observation.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
