import os
import json
import threading
import subprocess
import time
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB max upload

UPLOAD_FOLDER = "."
ALLOWED_EXTENSIONS = {"dxf"}

pipeline_status = {
    "running": False,
    "stage": None,
    "message": "",
    "done": False,
    "error": None,
    "progress": 0,
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def run_pipeline(dxf_path):
    global pipeline_status
    pipeline_status.update({"running": True, "done": False, "error": None, "progress": 0})

    try:
        pipeline_status.update({"stage": "Extracting wall faces", "message": "Running flatten_3dface.py...", "progress": 20})
        result = subprocess.run(
            ["python", "flatten_3dface.py"],
            capture_output=True, text=True, cwd="."
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout)

        extract_lines = [l for l in result.stdout.strip().splitlines() if l]
        pipeline_status.update({
            "stage": "Rasterizing grid",
            "message": f"{extract_lines[-1] if extract_lines else 'Wall faces extracted'}. Rasterizing...",
            "progress": 50
        })

        result2 = subprocess.run(
            ["python", "rasterize_to_grid.py"],
            capture_output=True, text=True, cwd="."
        )
        if result2.returncode != 0:
            raise RuntimeError(result2.stderr or result2.stdout)

        pipeline_status.update({
            "running": False,
            "done": True,
            "stage": "Complete",
            "message": "Pipeline finished successfully.",
            "progress": 100,
        })

    except Exception as e:
        pipeline_status.update({
            "running": False,
            "done": False,
            "error": str(e),
            "stage": "Error",
            "message": str(e),
            "progress": 0,
        })


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    results_ready = (
        os.path.exists("grid.npy")
        and os.path.exists("grid_meta.json")
        and os.path.exists("grid_preview.png")
    )
    dxf_exists = os.path.exists("house.dxf")

    meta = {}
    if os.path.exists("grid_meta.json"):
        try:
            with open("grid_meta.json") as f:
                meta = json.load(f)
        except Exception:
            pass

    return jsonify({
        "pipeline": pipeline_status,
        "results_ready": results_ready,
        "dxf_loaded": dxf_exists,
        "dxf_filename": "house.dxf" if dxf_exists else None,
        "meta": meta,
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Only .dxf files are allowed"}), 400

    filename = secure_filename(f.filename)
    save_path = "house.dxf"
    f.save(save_path)
    return jsonify({"success": True, "filename": filename})


@app.route("/api/run", methods=["POST"])
def api_run():
    global pipeline_status
    if pipeline_status["running"]:
        return jsonify({"error": "Pipeline already running"}), 409
    if not os.path.exists("house.dxf"):
        return jsonify({"error": "No DXF file loaded. Please upload one first."}), 400

    dxf_path = "house.dxf"
    thread = threading.Thread(target=run_pipeline, args=(dxf_path,), daemon=True)
    thread.start()
    return jsonify({"started": True})


@app.route("/api/preview")
def api_preview():
    if not os.path.exists("grid_preview.png"):
        return jsonify({"error": "No preview available"}), 404
    return send_file("grid_preview.png", mimetype="image/png")


@app.route("/api/download/grid")
def download_grid():
    if not os.path.exists("grid.npy"):
        return jsonify({"error": "No grid file available"}), 404
    return send_file("grid.npy", as_attachment=True, download_name="grid.npy")


@app.route("/api/download/meta")
def download_meta():
    if not os.path.exists("grid_meta.json"):
        return jsonify({"error": "No metadata file available"}), 404
    return send_file("grid_meta.json", as_attachment=True, download_name="grid_meta.json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
