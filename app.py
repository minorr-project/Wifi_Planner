# app.py
import os
import sys
import json
import threading
import subprocess
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

ALLOWED_EXTENSIONS = {"dxf"}

current_dxf = None

pipeline_status = {
    "running": False,
    "stage": None,
    "message": "",
    "done": False,
    "error": None,
    "progress": 0,
}

optimization_status = {
    "running": False,
    "stage": None,
    "message": "",
    "done": False,
    "error": None,
    "progress": 0,
    "strategy": None,
    "result": None,
}


def grid_results_ready():
    return os.path.exists("grid.npy") and os.path.exists("grid_meta.json")


def ensure_preview_from_grid(grid_path, preview_path, title):
    if os.path.exists(preview_path):
        return True
    if not os.path.exists(grid_path):
        return False

    import matplotlib.pyplot as plt

    grid = np.load(grid_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, origin="lower", cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(preview_path, dpi=200, bbox_inches="tight")
    plt.close()
    return True


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def run_pipeline(dxf_path):
    global pipeline_status
    pipeline_status.update({"running": True, "done": False, "error": None, "progress": 10})
    try:
        pipeline_status.update({
            "stage": "Processing DXF",
            "message": "Analysing entities and rasterizing floor plan…",
            "progress": 30,
        })
        result = subprocess.run(
            [sys.executable, "dxf_pipeline_general.py", dxf_path, "."],
            capture_output=True, text=True, cwd="."
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout)

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


def run_optimization_thread(strategy, num_routers):
    global optimization_status
    optimization_status.update({
        "running": True, "done": False, "error": None,
        "progress": 10, "strategy": strategy, "result": None,
    })
    try:
        optimization_status.update({
            "stage": "Optimizing",
            "message": f"Running {strategy} placement for {num_routers} router(s)…",
            "progress": 30,
        })
        from optimization import run_optimization
        result = run_optimization(strategy, num_routers)
        optimization_status.update({
            "running": False,
            "done": True,
            "stage": "Complete",
            "message": "Optimization finished.",
            "progress": 100,
            "result": result,
        })
    except Exception as e:
        optimization_status.update({
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
    results_ready = grid_results_ready()
    dxf_exists = current_dxf is not None and os.path.exists(current_dxf)
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
        "dxf_filename": os.path.basename(current_dxf) if current_dxf else None,
        "meta": meta,
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    global current_dxf
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Only .dxf files are allowed"}), 400
    filename = secure_filename(f.filename)
    f.save(filename)
    current_dxf = filename
    return jsonify({"success": True, "filename": filename})


@app.route("/api/run", methods=["POST"])
def api_run():
    global pipeline_status
    if pipeline_status["running"]:
        return jsonify({"error": "Pipeline already running"}), 409
    if not current_dxf or not os.path.exists(current_dxf):
        return jsonify({"error": "No DXF file loaded. Please upload one first."}), 400
    thread = threading.Thread(target=run_pipeline, args=(current_dxf,), daemon=True)
    thread.start()
    return jsonify({"started": True})


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    global optimization_status
    if optimization_status["running"]:
        return jsonify({"error": "Optimization already running"}), 409
    if not os.path.exists("grid.npy"):
        return jsonify({"error": "Grid not ready. Run the pipeline first."}), 400

    data = request.get_json(silent=True) or {}
    strategy = data.get("strategy", "ga")
    if strategy not in ("ga", "random", "uniform"):
        return jsonify({"error": "Invalid strategy. Choose ga, random, or uniform."}), 400
    try:
        num_routers = int(data.get("num_routers", 2))
        if num_routers < 1 or num_routers > 20:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({"error": "num_routers must be an integer between 1 and 20."}), 400

    optimization_status = {
        "running": False, "stage": None, "message": "",
        "done": False, "error": None, "progress": 0,
        "strategy": strategy, "result": None,
    }
    thread = threading.Thread(
        target=run_optimization_thread, args=(strategy, num_routers), daemon=True
    )
    thread.start()
    return jsonify({"started": True})


@app.route("/api/optimize/status")
def api_optimize_status():
    return jsonify(optimization_status)


@app.route("/api/optimize/image/<strategy>")
def api_optimize_image(strategy):
    if strategy not in ("ga", "random", "uniform"):
        return jsonify({"error": "Invalid strategy"}), 400
    path = os.path.join("outputs", "images", f"{strategy}_placement.png")
    if not os.path.exists(path):
        return jsonify({"error": "Image not available"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/api/preview")
def api_preview():
    path = "grid_display_preview.png" if os.path.exists("grid_display_preview.png") else "grid_preview.png"
    if not os.path.exists(path):
        return jsonify({"error": "No preview available"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/api/preview/opt")
def api_preview_opt():
    if not ensure_preview_from_grid("grid.npy", "grid_preview.png", "Optimization Grid"):
        return jsonify({"error": "No optimization preview available"}), 404
    return send_file("grid_preview.png", mimetype="image/png")


@app.route("/api/download/grid")
def download_grid():
    if not os.path.exists("grid.npy"):
        return jsonify({"error": "No grid file available"}), 404
    return send_file("grid.npy", as_attachment=True, download_name="grid.npy")


@app.route("/api/download/grid_display")
def download_grid_display():
    if not os.path.exists("grid_display.npy"):
        return jsonify({"error": "No display grid file available"}), 404
    return send_file("grid_display.npy", as_attachment=True, download_name="grid_display.npy")


@app.route("/api/download/meta")
def download_meta():
    if not os.path.exists("grid_meta.json"):
        return jsonify({"error": "No metadata file available"}), 404
    return send_file("grid_meta.json", as_attachment=True, download_name="grid_meta.json")


@app.route("/api/download/meta_display")
def download_meta_display():
    if not os.path.exists("grid_display_meta.json"):
        return jsonify({"error": "No display metadata file available"}), 404
    return send_file("grid_display_meta.json", as_attachment=True, download_name="grid_display_meta.json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
