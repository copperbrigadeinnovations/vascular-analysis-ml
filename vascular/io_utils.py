"""Image loading: DICOM (pydicom) or plain images (PNG/JPG/TIFF)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _is_dicom(path: Path) -> bool:
    if path.suffix.lower() in {".dcm", ".dicom"}:
        return True
    if path.suffix.lower() in _IMAGE_EXTS:
        return False
    try:
        with open(path, "rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except OSError:
        return False


def _normalize(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    return np.clip((img - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)


def _pick_frame(stack: np.ndarray) -> np.ndarray:
    if stack.shape[0] == 1:
        return stack[0]
    flat = stack.reshape(stack.shape[0], -1).astype(np.float32)
    return stack[int(np.argmax(flat.std(axis=1)))]


def load_image(path: str | Path) -> tuple[np.ndarray, tuple[float, float] | None, dict]:
    """Return (image scaled to [0,1], pixel spacing (row_mm, col_mm) or None, metadata)."""
    path = Path(path)
    meta = {"source": str(path), "format": "unknown"}
    spacing = None

    if _is_dicom(path):
        import pydicom

        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = arr[..., :3].mean(axis=-1)
        if arr.ndim == 3:
            arr = _pick_frame(arr)
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            arr = arr.max() - arr
        ps = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
        if ps is not None and len(ps) >= 2:
            spacing = (float(ps[0]), float(ps[1]))
        meta.update(
            format="DICOM",
            modality=str(getattr(ds, "Modality", "")),
            study=str(getattr(ds, "StudyDescription", "")),
            series=str(getattr(ds, "SeriesDescription", "")),
        )
    else:
        from skimage import io as skio

        arr = skio.imread(str(path))
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            from skimage.color import rgb2gray

            arr = rgb2gray(arr[..., :3])
        if arr.ndim == 3:
            arr = _pick_frame(arr)
        meta["format"] = path.suffix.lower().lstrip(".") or "image"

    if arr.ndim != 2:
        raise ValueError(f"unsupported image shape {arr.shape}")

    return _normalize(arr), spacing, meta
