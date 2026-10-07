"""Web front end for the vascular analysis pipeline (research use only).

Run:
    python app.py
Then open http://127.0.0.1:5000 and upload an image.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from run_analysis import EXTS, analyze_file

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = 25

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", error="Please choose an image file to upload.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in EXTS:
        return render_template(
            "index.html",
            error=f"Unsupported file type '{suffix or 'unknown'}'. Supported: {', '.join(sorted(EXTS))}",
        )

    spacing_raw = (request.form.get("spacing_mm") or "").strip()
    spacing_mm = None
    if spacing_raw:
        try:
            spacing_mm = float(spacing_raw)
        except ValueError:
            return render_template("index.html", error="Pixel spacing must be a number (mm/px).")

    token = uuid.uuid4().hex[:12]
    safe_name = secure_filename(file.filename) or "upload"
    saved_path = UPLOAD_DIR / f"{token}_{safe_name}"
    file.save(saved_path)

    try:
        paragraph, metrics, qual, recommendation = analyze_file(saved_path, UPLOAD_DIR, spacing_mm=spacing_mm)
    except Exception as exc:
        return render_template("index.html", error=f"Analysis failed: {exc}")

    overlay_name = f"{saved_path.stem}_overlay.png"
    rec_body = recommendation.removeprefix("Recommendations: ") if recommendation else ""
    return render_template(
        "index.html",
        observation=paragraph,
        recommendation=rec_body,
        quality=qual.get("label", "unknown"),
        filename=file.filename,
        overlay_url=url_for("static", filename=f"uploads/{overlay_name}"),
        n_branches=metrics.get("n_branches"),
        total_length=metrics.get("total_length"),
        n_stenoses=len(metrics.get("candidate_stenoses", [])),
    )


if __name__ == "__main__":
    app.run(debug=True)
