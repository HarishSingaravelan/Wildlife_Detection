import os
import io
import json
import base64
import threading
import traceback
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np
from PIL import Image, ImageFile, ImageOps

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

PORT = 8080
LOCK = threading.Lock()


# ==============================================================================
# AUTO-DISCOVER DIRECTORIES (Original 'csvs' is strictly READ-ONLY)
# ==============================================================================
def find_directories():
    search_roots = [
        "../../../../shared/rc/turbine/Caribou/COCO_Dataset",
        "../../../../shared/rc/turbine/Caribou",
        "./Caribou/COCO_Dataset",
        "./COCO_Dataset",
        "./Caribou",
        ".",
    ]
    csv_dir, tif_dir, img_dir = None, None, None

    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if csv_dir is None and os.path.basename(dirpath) == "csvs":
                if any(f.endswith(".csv") for f in filenames):
                    csv_dir = dirpath
            if tif_dir is None and os.path.basename(dirpath) == "tif":
                tif_dir = dirpath
            if img_dir is None and os.path.basename(dirpath) == "images" and "COCO_Dataset" in dirpath:
                img_dir = dirpath

    raw_csv_dir = csv_dir or "./csvs"
    # New output folder right next to 'csvs' -> 'csvs_fixed'
    fixed_csv_dir = os.path.join(os.path.dirname(os.path.abspath(raw_csv_dir)), "csvs_fixed")
    os.makedirs(fixed_csv_dir, exist_ok=True)

    return raw_csv_dir, fixed_csv_dir, tif_dir or "./tif", img_dir or "./images"


RAW_CSV_DIR, FIXED_CSV_DIR, TIF_DIR, IMG_DIR = find_directories()
TIF_CACHE = {"filename": None, "image": None, "path": None}


def resolve_image_path(filename):
    """Finds the mosaic file across tif/ and images/ directories."""
    candidates = [
        os.path.join(TIF_DIR, filename),
        os.path.join(IMG_DIR, filename),
        os.path.join(os.path.dirname(RAW_CSV_DIR), "tif", filename),
        os.path.join(os.path.dirname(RAW_CSV_DIR), "images", filename),
    ]
    stem = os.path.splitext(filename)[0]
    for ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
        candidates.append(os.path.join(TIF_DIR, stem + ext))
        candidates.append(os.path.join(IMG_DIR, stem + ext))

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Could not find '{filename}' in:\n  - {os.path.abspath(TIF_DIR)}\n  - {os.path.abspath(IMG_DIR)}"
    )


def get_cropped_patch(filename, cx, cy, size, auto_enhance=False):
    """Crops only the requested window before converting to 8-bit RGB."""
    with LOCK:
        if TIF_CACHE["filename"] != filename or TIF_CACHE["image"] is None:
            img_path = resolve_image_path(filename)
            print(f"[INFO] Opening mosaic: {img_path}")
            img = Image.open(img_path)
            TIF_CACHE["filename"] = filename
            TIF_CACHE["image"] = img
            TIF_CACHE["path"] = img_path
        else:
            img = TIF_CACHE["image"]

        half = size // 2
        x1 = max(0, min(img.width - size, cx - half))
        y1 = max(0, min(img.height - size, cy - half))
        x2 = min(img.width, x1 + size)
        y2 = min(img.height, y1 + size)

        patch = img.crop((x1, y1, x2, y2))
        arr = np.array(patch)

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        p_min, p_max = np.percentile(arr, (1, 99))
        if p_max > p_min:
            arr = np.clip((arr - p_min) / (p_max - p_min) * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] == 4:
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        rgb = arr[:, :, :3].astype(np.float32)
        arr = (rgb * alpha + 30.0 * (1.0 - alpha)).astype(np.uint8)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]

    rgb_patch = Image.fromarray(arr, mode="RGB")
    if auto_enhance:
        rgb_patch = ImageOps.autocontrast(rgb_patch, cutoff=1)

    buf = io.BytesIO()
    rgb_patch.save(buf, format="JPEG", quality=92)
    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "offset_x": int(x1),
        "offset_y": int(y1),
        "crop_w": int(x2 - x1),
        "crop_h": int(y2 - y1),
        "image_b64": f"data:image/jpeg;base64,{b64_img}",
    }


def _read_csv_file(path):
    headers, meta, rows = [], {}, []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                headers.append(stripped)
                if ":" in stripped:
                    k, v = stripped[1:].split(":", 1)
                    meta[k.strip()] = v.strip()
            elif stripped.lower().startswith("x1,y1"):
                continue
            else:
                parts = [int(float(x)) for x in stripped.split(",")]
                if len(parts) == 4:
                    rows.append(parts)
    return headers, meta, rows


def parse_csv_file(csv_name):
    """
    Reads original 1D lines from RAW_CSV_DIR (read-only) and loads saved
    2D progress from FIXED_CSV_DIR if it exists.
    """
    raw_path = os.path.join(RAW_CSV_DIR, csv_name)
    fixed_path = os.path.join(FIXED_CSV_DIR, csv_name)

    headers, meta, orig_rows = _read_csv_file(raw_path)

    if os.path.exists(fixed_path):
        _, _, current_rows = _read_csv_file(fixed_path)
        while len(current_rows) < len(orig_rows):
            current_rows.append(list(orig_rows[len(current_rows)]))
    else:
        current_rows = [list(r) for r in orig_rows]

    return headers, meta, current_rows, orig_rows


def write_csv_file(csv_name, headers, rows):
    """Writes ONLY to FIXED_CSV_DIR, leaving RAW_CSV_DIR untouched."""
    os.makedirs(FIXED_CSV_DIR, exist_ok=True)
    fixed_path = os.path.join(FIXED_CSV_DIR, csv_name)
    with open(fixed_path, "w", encoding="utf-8") as f:
        for h in headers:
            f.write(f"{h}\n")
        f.write("x1,y1,x2,y2\n")
        for r in rows:
            f.write(f"{int(r[0])},{int(r[1])},{int(r[2])},{int(r[3])}\n")
    return fixed_path


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Caribou 1D-to-2D Box Annotator</title>
<style>
  * { box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }
  body { margin: 0; display: flex; height: 100vh; background: #121212; color: #e0e0e0; overflow: hidden; }
  #sidebar { width: 340px; background: #1e1e1e; border-right: 1px solid #333; display: flex; flex-direction: column; padding: 14px; }
  #main { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; }
  h2 { margin: 0 0 6px 0; font-size: 16px; color: #fff; }
  .save-path-info { font-size: 11px; color: #34d399; background: #064e3b; padding: 5px 8px; border-radius: 4px; margin-bottom: 10px; word-break: break-all; }
  select, button, input { background: #2c2c2c; color: #fff; border: 1px solid #444; padding: 7px 10px; border-radius: 5px; font-size: 13px; }
  button { cursor: pointer; font-weight: 600; transition: 0.15s; }
  button:hover { background: #3d3d3d; }
  button.primary { background: #2563eb; border-color: #3b82f6; }
  button.success { background: #059669; border-color: #10b981; }
  .toolbar { display: flex; gap: 8px; align-items: center; background: #1e1e1e; padding: 10px 14px; border-radius: 8px; border: 1px solid #333; margin-bottom: 10px; flex-wrap: wrap; }
  #row-list { flex: 1; overflow-y: auto; margin-top: 10px; border: 1px solid #333; border-radius: 5px; background: #161616; }
  .row-item { padding: 7px 10px; border-bottom: 1px solid #262626; cursor: pointer; font-size: 12px; display: flex; justify-content: space-between; }
  .row-item:hover { background: #262626; }
  .row-item.active { background: #1e3a8a; color: #fff; font-weight: bold; }
  .badge-done { color: #34d399; font-weight: bold; }
  .badge-todo { color: #fbbf24; font-weight: bold; }
  #canvas-container { position: relative; border: 2px solid #444; border-radius: 4px; background: #000; }
  canvas { display: block; cursor: crosshair; }
  .status-bar { margin-top: 10px; font-size: 13px; color: #aaa; display: flex; gap: 18px; }
  .progress-box { background: #262626; border-radius: 4px; height: 10px; width: 100%; margin: 8px 0; overflow: hidden; }
  #progress-fill { background: #10b981; height: 100%; width: 0%; transition: width 0.2s; }
  #error-banner { display: none; background: #7f1d1d; color: #fecaca; padding: 10px 14px; border-radius: 6px; margin-bottom: 10px; max-width: 650px; font-size: 12px; white-space: pre-wrap; border: 1px solid #ef4444; }
</style>
</head>
<body>

<div id="sidebar">
  <h2>🦌 Caribou CSV Annotator</h2>
  <div class="save-path-info" id="out-dir-banner">Saving to: csvs_fixed/ (Originals untouched)</div>
  <label style="font-size:12px; color:#aaa;">Select CSV File:</label>
  <select id="csv-select" onchange="loadCSV(this.value)" style="margin: 5px 0 8px 0; width: 100%;"></select>

  <div style="display:flex; justify-content:space-between; font-size:12px;">
    <span id="progress-text">0 / 0 Fixed</span>
    <span id="meta-label" style="color:#60a5fa;"></span>
  </div>
  <div class="progress-box"><div id="progress-fill"></div></div>

  <button class="primary" style="margin-bottom:6px;" onclick="jumpNextUnfixed()">⏭ Resume Next Unfixed (N)</button>
  <div id="row-list"></div>
</div>

<div id="main">
  <div id="error-banner"></div>

  <div class="toolbar">
    <button onclick="prevRow()">⬅ Prev (A)</button>
    <span id="current-idx-label" style="font-weight:bold; min-width:90px; text-align:center;">Row 0 / 0</span>
    <button onclick="nextRow()">Next (D) ➡</button>
    <span style="color:#444;">|</span>
    <label style="font-size:13px;">Crop:
      <select id="crop-size" onchange="renderCurrentRow()">
        <option value="256">256x256 px</option>
        <option value="400" selected>400x400 px</option>
        <option value="600">600x600 px</option>
        <option value="1000">1000x1000 px</option>
      </select>
    </label>
    <label style="font-size:13px; cursor:pointer;">
      <input type="checkbox" id="auto-enhance" onchange="renderCurrentRow()"> Boost Contrast
    </label>
    <label style="font-size:13px;">Brightness:
      <input type="range" id="brightness" min="100" max="300" value="100" style="width:70px; vertical-align:middle;" oninput="updateBrightness()">
    </label>
    <label style="font-size:13px; cursor:pointer;">
      <input type="checkbox" id="auto-advance" checked> Auto-Advance
    </label>
    <button onclick="resetCurrentRow()">↺ Reset (R)</button>
    <button class="success" onclick="saveFullCSV()">💾 Save to csvs_fixed (S)</button>
  </div>

  <div id="canvas-container">
    <canvas id="annot-canvas" width="650" height="650"></canvas>
  </div>

  <div class="status-bar">
    <span><b style="color:#ff4444;">━━━</b> Original 1D Line</span>
    <span><b style="color:#00ff88;">───</b> Saved 2D Box</span>
    <span><b style="color:#facc15;">- - -</b> Nearby Row Boxes</span>
    <span id="coord-readout">Coords: -</span>
  </div>
</div>

<script>
let currentCSV = "";
let rows = [];
let origRows = [];
let meta = {};
let currentIndex = 0;

let cropInfo = { offset_x: 0, offset_y: 0, crop_w: 400, crop_h: 400 };
let currentImg = new Image();
let imgLoaded = false;
let isDragging = false;
let dragStart = { x: 0, y: 0 };
let dragEnd = { x: 0, y: 0 };

const canvas = document.getElementById("annot-canvas");
const ctx = canvas.getContext("2d");

function updateBrightness() {
  const val = document.getElementById("brightness").value;
  canvas.style.filter = `brightness(${val}%)`;
}

function isRowFixed(r) {
  return Math.abs(r[3] - r[1]) > 1 && Math.abs(r[2] - r[0]) > 1;
}

async function init() {
  const res = await fetch("/api/csvs");
  const data = await res.json();
  if (data.fixed_dir) {
    document.getElementById("out-dir-banner").textContent = `Output Dir: ${data.fixed_dir}`;
  }
  const sel = document.getElementById("csv-select");
  sel.innerHTML = "";
  data.csvs.forEach(c => {
    const opt = document.createElement("option");
    opt.value = c.name;
    opt.textContent = `${c.name} (${c.fixed}/${c.total})`;
    sel.appendChild(opt);
  });
  if (data.csvs.length > 0) loadCSV(data.csvs[0].name);
}

async function loadCSV(name) {
  currentCSV = name;
  const res = await fetch(`/api/load_csv?name=${encodeURIComponent(name)}`);
  const data = await res.json();
  rows = data.rows;
  origRows = data.orig_rows;
  meta = data.meta;
  document.getElementById("meta-label").textContent = meta.label || "";
  currentIndex = 0;
  updateSidebar();
  jumpNextUnfixed(false);
}

function updateSidebar() {
  const list = document.getElementById("row-list");
  list.innerHTML = "";
  let fixedCount = 0;

  rows.forEach((r, idx) => {
    const fixed = isRowFixed(r);
    if (fixed) fixedCount++;
    const div = document.createElement("div");
    div.className = "row-item" + (idx === currentIndex ? " active" : "");
    div.id = `row-item-${idx}`;
    const h = Math.abs(r[3] - r[1]);
    div.innerHTML = `
      <span>#${idx + 1}: [${r[0]}, ${r[1]}, ${r[2]}, ${r[3]}]</span>
      <span class="${fixed ? 'badge-done' : 'badge-todo'}">${fixed ? '2D (h='+h+') ✓' : '1D Line ⚠'}</span>
    `;
    div.onclick = () => { currentIndex = idx; renderCurrentRow(); };
    list.appendChild(div);
  });

  const pct = rows.length ? Math.round((fixedCount / rows.length) * 100) : 0;
  document.getElementById("progress-text").textContent = `${fixedCount} / ${rows.length} Fixed (${pct}%)`;
  document.getElementById("progress-fill").style.width = `${pct}%`;

  const activeEl = document.getElementById(`row-item-${currentIndex}`);
  if (activeEl) activeEl.scrollIntoView({ block: "nearest" });
}

async function renderCurrentRow() {
  if (!rows.length) return;
  document.getElementById("current-idx-label").textContent = `Row ${currentIndex + 1} / ${rows.length}`;
  updateSidebar();

  const cropSize = parseInt(document.getElementById("crop-size").value, 10);
  const enhance = document.getElementById("auto-enhance").checked ? 1 : 0;
  const orig = origRows[currentIndex];
  const cx = Math.round((orig[0] + orig[2]) / 2);
  const cy = Math.round((orig[1] + orig[3]) / 2);

  cropInfo = { offset_x: cx - Math.floor(cropSize/2), offset_y: cy - Math.floor(cropSize/2), crop_w: cropSize, crop_h: cropSize };
  imgLoaded = false;
  drawCanvas();

  const res = await fetch(`/api/get_crop?name=${encodeURIComponent(currentCSV)}&cx=${cx}&cy=${cy}&size=${cropSize}&enhance=${enhance}`);
  const data = await res.json();

  const errBanner = document.getElementById("error-banner");
  if (data.error) {
    errBanner.style.display = "block";
    errBanner.textContent = "Backend Error: " + data.error;
    drawCanvas();
    return;
  }
  errBanner.style.display = "none";

  cropInfo = { offset_x: data.offset_x, offset_y: data.offset_y, crop_w: data.crop_w, crop_h: data.crop_h };
  currentImg.onload = () => {
    imgLoaded = true;
    drawCanvas();
  };
  currentImg.src = data.image_b64;
}

function globalToCanvas(gx, gy) {
  const scaleX = canvas.width / cropInfo.crop_w;
  const scaleY = canvas.height / cropInfo.crop_h;
  return {
    x: (gx - cropInfo.offset_x) * scaleX,
    y: (gy - cropInfo.offset_y) * scaleY
  };
}

function canvasToGlobal(cx, cy) {
  const scaleX = cropInfo.crop_w / canvas.width;
  const scaleY = cropInfo.crop_h / canvas.height;
  return {
    x: Math.round(cropInfo.offset_x + cx * scaleX),
    y: Math.round(cropInfo.offset_y + cy * scaleY)
  };
}

function drawCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (imgLoaded) {
    ctx.drawImage(currentImg, 0, 0, canvas.width, canvas.height);
  }

  rows.forEach((r, idx) => {
    if (idx === currentIndex) return;
    if (r[2] >= cropInfo.offset_x && r[0] <= cropInfo.offset_x + cropInfo.crop_w &&
        r[3] >= cropInfo.offset_y && r[1] <= cropInfo.offset_y + cropInfo.crop_h) {
      const p1 = globalToCanvas(r[0], r[1]);
      const p2 = globalToCanvas(r[2], r[3]);
      ctx.save();
      ctx.strokeStyle = "#facc15";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      if (isRowFixed(r)) {
        ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
      } else {
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
      }
      ctx.restore();
    }
  });

  const orig = origRows[currentIndex];
  const o1 = globalToCanvas(orig[0], orig[1]);
  const o2 = globalToCanvas(orig[2], orig[3]);
  ctx.save();
  ctx.strokeStyle = "#ff3333";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(o1.x, o1.y);
  ctx.lineTo(o2.x, o2.y);
  ctx.stroke();
  ctx.fillStyle = "#00ffff";
  ctx.beginPath(); ctx.arc(o1.x, o1.y, 4, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(o2.x, o2.y, 4, 0, Math.PI * 2); ctx.fill();
  ctx.restore();

  const cur = rows[currentIndex];
  if (isRowFixed(cur)) {
    const c1 = globalToCanvas(cur[0], cur[1]);
    const c2 = globalToCanvas(cur[2], cur[3]);
    ctx.save();
    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = 2.5;
    ctx.strokeRect(c1.x, c1.y, c2.x - c1.x, c2.y - c1.y);
    ctx.fillStyle = "rgba(0, 255, 136, 0.12)";
    ctx.fillRect(c1.x, c1.y, c2.x - c1.x, c2.y - c1.y);
    ctx.restore();
  }

  if (isDragging) {
    const x = Math.min(dragStart.x, dragEnd.x);
    const y = Math.min(dragStart.y, dragEnd.y);
    const w = Math.abs(dragEnd.x - dragStart.x);
    const h = Math.abs(dragEnd.y - dragStart.y);
    ctx.save();
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = "rgba(56, 189, 248, 0.2)";
    ctx.fillRect(x, y, w, h);
    ctx.restore();
  }
}

canvas.addEventListener("mousedown", e => {
  const rect = canvas.getBoundingClientRect();
  isDragging = true;
  dragStart = { x: e.clientX - rect.left, y: e.clientY - rect.top };
  dragEnd = { ...dragStart };
});

canvas.addEventListener("mousemove", e => {
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const g = canvasToGlobal(cx, cy);
  document.getElementById("coord-readout").textContent = `Global TIF Coords: (${g.x}, ${g.y})`;
  if (!isDragging) return;
  dragEnd = { x: cx, y: cy };
  drawCanvas();
});

canvas.addEventListener("mouseup", async () => {
  if (!isDragging) return;
  isDragging = false;

  const g1 = canvasToGlobal(Math.min(dragStart.x, dragEnd.x), Math.min(dragStart.y, dragEnd.y));
  const g2 = canvasToGlobal(Math.max(dragStart.x, dragEnd.x), Math.max(dragStart.y, dragEnd.y));

  if (Math.abs(g2.x - g1.x) < 3 || Math.abs(g2.y - g1.y) < 3) {
    drawCanvas();
    return;
  }

  const newBox = [g1.x, g1.y, g2.x, g2.y];
  rows[currentIndex] = newBox;
  drawCanvas();

  await fetch("/api/update_row", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: currentCSV, index: currentIndex, box: newBox })
  });

  updateSidebar();

  if (document.getElementById("auto-advance").checked) {
    setTimeout(() => jumpNextUnfixed(true), 150);
  }
});

function prevRow() { if (currentIndex > 0) { currentIndex--; renderCurrentRow(); } }
function nextRow() { if (currentIndex < rows.length - 1) { currentIndex++; renderCurrentRow(); } }

function jumpNextUnfixed(fromCurrent = true) {
  const start = fromCurrent ? currentIndex + 1 : 0;
  for (let i = start; i < rows.length; i++) {
    if (!isRowFixed(rows[i])) { currentIndex = i; renderCurrentRow(); return; }
  }
  for (let i = 0; i < start; i++) {
    if (!isRowFixed(rows[i])) { currentIndex = i; renderCurrentRow(); return; }
  }
  renderCurrentRow();
}

async function resetCurrentRow() {
  rows[currentIndex] = [...origRows[currentIndex]];
  await fetch("/api/update_row", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: currentCSV, index: currentIndex, box: rows[currentIndex] })
  });
  renderCurrentRow();
}

async function saveFullCSV() {
  const res = await fetch("/api/save_csv", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: currentCSV, rows: rows })
  });
  const data = await res.json();
  alert(`Saved to:\n${data.saved_path}\n\n(Original file in csvs/ was NOT modified)`);
}

window.addEventListener("keydown", e => {
  if (e.key === "a" || e.key === "ArrowLeft") prevRow();
  if (e.key === "d" || e.key === "ArrowRight" || e.key === " ") nextRow();
  if (e.key === "n" || e.key === "N") jumpNextUnfixed(true);
  if (e.key === "r" || e.key === "R") resetCurrentRow();
  if (e.key === "s" || e.key === "S") saveFullCSV();
});

init();
</script>
</body>
</html>
"""


class AnnotatorHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif parsed.path == "/api/csvs":
            csv_files = sorted([f for f in os.listdir(RAW_CSV_DIR) if f.endswith(".csv")])
            summary = []
            for f in csv_files:
                _, _, r, _ = parse_csv_file(f)
                fixed = sum(1 for b in r if abs(b[3] - b[1]) > 1 and abs(b[2] - b[0]) > 1)
                summary.append({"name": f, "total": len(r), "fixed": fixed})
            self._send_json({"csvs": summary, "fixed_dir": os.path.abspath(FIXED_CSV_DIR)})

        elif parsed.path == "/api/load_csv":
            name = qs["name"][0]
            headers, meta, rows, orig_rows = parse_csv_file(name)
            self._send_json({"headers": headers, "meta": meta, "rows": rows, "orig_rows": orig_rows})

        elif parsed.path == "/api/get_crop":
            try:
                name = qs["name"][0]
                cx, cy, size = int(qs["cx"][0]), int(qs["cy"][0]), int(qs["size"][0])
                enhance = qs.get("enhance", ["0"])[0] == "1"
                _, meta, _, _ = parse_csv_file(name)
                result = get_cropped_patch(meta["filename"], cx, cy, size, auto_enhance=enhance)
                self._send_json(result)
            except Exception as e:
                traceback.print_exc()
                self._send_json({"error": str(e)})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))

        if self.path == "/api/update_row":
            name = body["name"]
            idx = int(body["index"])
            box = body["box"]
            headers, _, rows, _ = parse_csv_file(name)
            rows[idx] = box
            saved_path = write_csv_file(name, headers, rows)
            self._send_json({"status": "ok", "saved_path": saved_path})

        elif self.path == "/api/save_csv":
            name = body["name"]
            rows = body["rows"]
            headers, _, _, _ = parse_csv_file(name)
            saved_path = write_csv_file(name, headers, rows)
            self._send_json({"status": "ok", "saved_path": saved_path})

    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


if __name__ == "__main__":
    print("=" * 75)
    print(f"Starting Caribou Annotator at http://localhost:{PORT}")
    print(f"  [READ-ONLY Original CSVs] : {os.path.abspath(RAW_CSV_DIR)}")
    print(f"  [NEW Saved 2D CSVs]       : {os.path.abspath(FIXED_CSV_DIR)}")
    print(f"  [TIF Mosaics Directory]   : {os.path.abspath(TIF_DIR)}")
    print("=" * 75)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), AnnotatorHandler)
    server.serve_forever()