"""
Qbit - Brain Tumor Segmentation Suite
=====================================
Professional Streamlit UI delegating inference to api.py, with rich
visualizations: animated tumor highlight, contour pulse, heatmap,
zoom view, risk gauge, and downloadable clinical report.

Run:
    streamlit run app.py
"""

import os
import sys
import time
import base64
import subprocess
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Qbit | Brain Tumor Segmentation",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS - Qbit Quantum Theme
# ============================================================
st.markdown(
    """
    <style>
    /* ---------- Background ---------- */
    .stApp {
        background:
            radial-gradient(1000px 600px at 0% 0%, rgba(34,211,238,0.10), transparent 60%),
            radial-gradient(900px 500px at 100% 0%, rgba(168,85,247,0.10), transparent 60%),
            radial-gradient(700px 400px at 50% 100%, rgba(236,72,153,0.06), transparent 60%),
            linear-gradient(180deg, #050814 0%, #0a0f1f 100%);
        background-attachment: fixed;
    }
    /* Animated quantum grid */
    .stApp::before {
        content: "";
        position: fixed; inset: 0;
        background-image:
            linear-gradient(rgba(34,211,238,0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(34,211,238,0.05) 1px, transparent 1px);
        background-size: 50px 50px;
        animation: grid-shift 20s linear infinite;
        pointer-events: none;
        z-index: 0;
    }
    @keyframes grid-shift {
        0%   { transform: translate(0, 0); }
        100% { transform: translate(50px, 50px); }
    }
    .main .block-container { position: relative; z-index: 1; }
    #MainMenu, footer, header { visibility: hidden; }

    /* ---------- Typography ---------- */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    h1 {
        background: linear-gradient(90deg, #22d3ee 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.03em;
        font-size: 2.6rem !important;
    }
    h2, h3, h4 { color: #e5e7eb !important; font-weight: 600 !important; letter-spacing: -0.01em; }
    p, .stMarkdown, label, .stCaption { color: #cbd5e1 !important; }

    /* ---------- Brand Bar ---------- */
    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.4rem;
        background: rgba(10, 15, 31, 0.7);
        border: 1px solid rgba(34, 211, 238, 0.18);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(34, 211, 238, 0.08);
    }
    .brand-left { display: flex; align-items: center; gap: 0.9rem; }
    .qbit-logo {
        width: 46px; height: 46px;
        border-radius: 12px;
        background: linear-gradient(135deg, #22d3ee 0%, #a855f7 100%);
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; color: white; font-size: 1.35rem;
        font-family: 'Courier New', monospace;
        box-shadow:
            0 8px 24px rgba(34, 211, 238, 0.4),
            inset 0 1px 0 rgba(255,255,255,0.2);
        position: relative;
    }
    .qbit-logo::after {
        content: "";
        position: absolute; inset: -3px;
        border-radius: 14px;
        background: linear-gradient(135deg, #22d3ee, #a855f7, #ec4899);
        z-index: -1;
        filter: blur(8px);
        opacity: 0.5;
    }
    .brand-name {
        color: #f1f5f9; font-weight: 700; font-size: 1.15rem;
        letter-spacing: -0.01em;
        background: linear-gradient(90deg, #22d3ee, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .brand-sub  { color: #64748b; font-size: 0.78rem; letter-spacing: 0.02em; }

    /* ---------- Cards ---------- */
    .info-card {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(34, 211, 238, 0.18);
        border-radius: 14px;
        padding: 1.4rem 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(8px);
    }
    .metric-card {
        background: linear-gradient(180deg, rgba(15,23,42,0.75) 0%, rgba(15,23,42,0.45) 100%);
        border: 1px solid rgba(34, 211, 238, 0.18);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        height: 100%;
        transition: border-color .25s ease, transform .25s ease, box-shadow .25s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: "";
        position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, #22d3ee, #a855f7, transparent);
        opacity: 0.4;
    }
    .metric-card:hover {
        border-color: rgba(34,211,238,0.5);
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(34,211,238,0.12);
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }
    .metric-value {
        color: #22d3ee;
        font-size: 1.5rem;
        font-weight: 700;
        word-break: break-all;
        line-height: 1.2;
    }
    .metric-value.warn   { color: #f59e0b; }
    .metric-value.danger { color: #ef4444; }
    .metric-value.ok     { color: #10b981; }
    .metric-value.purple { color: #a855f7; }

    /* ---------- Buttons ---------- */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(90deg, #22d3ee 0%, #a855f7 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.6rem;
        font-weight: 600;
        transition: all 0.25s ease;
        width: 100%;
        letter-spacing: 0.01em;
        position: relative;
        overflow: hidden;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(168, 85, 247, 0.4);
    }
    .stButton > button:active { transform: translateY(0); }

    /* ---------- File Uploader ---------- */
    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.5);
        border: 2px dashed rgba(34, 211, 238, 0.35);
        border-radius: 14px;
        padding: 0.5rem;
        transition: all .25s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(168, 85, 247, 0.6);
        background: rgba(15, 23, 42, 0.7);
    }
    [data-testid="stFileUploader"] section { background: transparent; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: rgba(5, 8, 20, 0.95);
        border-right: 1px solid rgba(34, 211, 238, 0.12);
    }
    [data-testid="stSidebar"] h3 {
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8 !important;
        margin-top: 1.2rem !important;
    }

    /* ---------- Badges ---------- */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-success { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
    .badge-info    { background: rgba(34, 211, 238, 0.15); color: #22d3ee; border: 1px solid rgba(34,211,238,0.3); }
    .badge-warning { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
    .badge-danger  { background: rgba(239, 68, 68, 0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
    .badge-purple  { background: rgba(168, 85, 247, 0.15); color: #a855f7; border: 1px solid rgba(168,85,247,0.3); }
    .pulse-dot {
        width: 8px; height: 8px;
        border-radius: 999px;
        background: currentColor;
        box-shadow: 0 0 0 0 currentColor;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 currentColor; }
        70%  { box-shadow: 0 0 0 8px transparent; }
        100% { box-shadow: 0 0 0 0 transparent; }
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 10px;
        padding: 0.6rem 1.15rem;
        color: #94a3b8;
        border: 1px solid rgba(34,211,238,0.12);
        transition: all .2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { border-color: rgba(34,211,238,0.4); color: #cbd5e1; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(34,211,238,0.2), rgba(168,85,247,0.2)) !important;
        color: #f1f5f9 !important;
        border-color: rgba(168,85,247,0.5) !important;
        box-shadow: 0 4px 16px rgba(34,211,238,0.15);
    }

    /* ---------- Images ---------- */
    [data-testid="stImage"] img {
        border-radius: 12px;
        border: 1px solid rgba(34, 211, 238, 0.25);
        box-shadow: 0 16px 40px rgba(0,0,0,0.45);
    }

    /* ---------- Section Headers ---------- */
    .section-num {
        display: inline-flex;
        width: 30px; height: 30px;
        border-radius: 9px;
        background: linear-gradient(135deg, #22d3ee, #a855f7);
        color: white; font-weight: 700;
        align-items: center; justify-content: center;
        margin-right: 0.7rem;
        font-size: 0.9rem;
        box-shadow: 0 4px 12px rgba(168,85,247,0.3);
    }
    .section-title {
        display: flex; align-items: center;
        font-size: 1.2rem; font-weight: 600;
        color: #f1f5f9; margin: 1.8rem 0 0.9rem;
    }

    hr { border-color: rgba(34, 211, 238, 0.12) !important; margin: 1.8rem 0 !important; }

    /* ---------- Sliders ---------- */
    [data-testid="stSlider"] [role="slider"] { background: #22d3ee !important; }

    /* ---------- Risk Gauge container ---------- */
    .gauge-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(180deg, rgba(15,23,42,0.75), rgba(15,23,42,0.4));
        border: 1px solid rgba(34, 211, 238, 0.2);
        border-radius: 14px;
        padding: 1.2rem;
        height: 100%;
    }

    /* ---------- Footer ---------- */
    .qbit-footer {
        text-align: center;
        margin-top: 2.5rem;
        padding: 1.2rem;
        color: #64748b;
        font-size: 0.85rem;
        border-top: 1px solid rgba(34, 211, 238, 0.1);
    }
    .qbit-footer span {
        background: linear-gradient(90deg, #22d3ee, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Constants
# ============================================================
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)
API_SCRIPT = "api.py"

# ============================================================
# Inference helpers
# ============================================================
def find_predicted_output(input_path: Path) -> Optional[Path]:
    candidates = [
        input_path.with_name(f"{input_path.stem}_predicted.png"),
        input_path.with_name(f"{input_path.stem}_predicted{input_path.suffix}"),
        input_path.with_name(f"{input_path.stem}_mask.png"),
        input_path.with_name(f"{input_path.stem}_output.png"),
        Path(f"{input_path.stem}_predicted.png"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run_inference(input_path: Path) -> Tuple[Optional[Path], Optional[str], float]:
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, API_SCRIPT, "--file", str(input_path)],
            capture_output=True, text=True, timeout=180,
        )
        elapsed = time.time() - start
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip() or "Unknown error from api.py"
            return None, err, elapsed
        output = find_predicted_output(input_path)
        if output is None:
            return None, "Inference completed but output file was not found.", elapsed
        return output, None, elapsed
    except subprocess.TimeoutExpired:
        return None, "Inference timed out (>180s).", time.time() - start
    except Exception as e:
        return None, str(e), time.time() - start


# ============================================================
# Visualization helpers
# ============================================================
def load_mask_as_binary(mask_path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    m = Image.open(mask_path).convert("L").resize(target_size, Image.NEAREST)
    arr = np.array(m)
    binary = (arr > 30).astype(np.uint8) * 255
    return binary


def colorize_mask(binary: np.ndarray, color=(239, 68, 68)) -> Image.Image:
    h, w = binary.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = binary
    return Image.fromarray(rgba, mode="RGBA")


def make_overlay(original: Image.Image, binary: np.ndarray, opacity: float, color=(239, 68, 68)) -> Image.Image:
    base = original.convert("RGBA")
    h, w = binary.shape

    # Glow layer (blurred mask, behind everything for halo effect)
    glow_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    glow_rgba[..., 0] = color[0]
    glow_rgba[..., 1] = color[1]
    glow_rgba[..., 2] = color[2]
    glow_rgba[..., 3] = (binary > 0).astype(np.uint8) * 120
    glow_layer = Image.fromarray(glow_rgba, mode="RGBA").filter(ImageFilter.GaussianBlur(radius=12))

    # Translucent fill
    fill_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    fill_rgba[..., 0] = color[0]
    fill_rgba[..., 1] = color[1]
    fill_rgba[..., 2] = color[2]
    fill_rgba[..., 3] = (binary > 0).astype(np.uint8) * int(255 * opacity)
    fill_layer = Image.fromarray(fill_rgba, mode="RGBA")

    # Bright contour (yellow)
    mask_img = Image.fromarray(binary, mode="L")
    edges = mask_img.filter(ImageFilter.FIND_EDGES)
    edges_thick = (np.array(edges) > 0).astype(np.uint8) * 255
    contour_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    contour_rgba[..., 0] = 250
    contour_rgba[..., 1] = 204
    contour_rgba[..., 2] = 21
    contour_rgba[..., 3] = edges_thick
    contour_layer = Image.fromarray(contour_rgba, mode="RGBA").filter(ImageFilter.MaxFilter(3))

    out = Image.alpha_composite(base, glow_layer)
    out = Image.alpha_composite(out, fill_layer)
    out = Image.alpha_composite(out, contour_layer)
    return out.convert("RGB")


def find_probability_map(input_path: Path) -> Optional[Path]:
    """Look for raw probability outputs from api.py (real model confidence)."""
    candidates = [
        input_path.with_name(f"{input_path.stem}_prob.npy"),
        input_path.with_name(f"{input_path.stem}_proba.npy"),
        input_path.with_name(f"{input_path.stem}_confidence.npy"),
        input_path.with_name(f"{input_path.stem}_prob.png"),
        input_path.with_name(f"{input_path.stem}_proba.png"),
        input_path.with_name(f"{input_path.stem}_confidence.png"),
        Path(f"{input_path.stem}_prob.npy"),
        Path(f"{input_path.stem}_prob.png"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_probability_map(prob_path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    """Load raw model probabilities as float32 array in [0, 1]."""
    if prob_path.suffix.lower() == ".npy":
        arr = np.load(str(prob_path)).astype(np.float32)
        # Squeeze to 2D and normalize if logits
        arr = np.squeeze(arr)
        if arr.ndim > 2:
            arr = arr[..., 0] if arr.shape[-1] == 1 else arr.mean(axis=-1)
        if arr.max() > 1.0 + 1e-3 or arr.min() < -1e-3:
            # Looks like logits, apply sigmoid
            arr = 1.0 / (1.0 + np.exp(-arr))
        # Resize via PIL
        pil = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
        pil = pil.resize(target_size, Image.BILINEAR)
        return np.array(pil).astype(np.float32) / 255.0
    else:
        pil = Image.open(prob_path).convert("L").resize(target_size, Image.BILINEAR)
        return np.array(pil).astype(np.float32) / 255.0


def estimate_confidence_from_mask(binary: np.ndarray) -> np.ndarray:
    """Fallback: derive a soft confidence-like field by blurring the binary mask.
    Pixels deep inside tumor get high values, near boundary get mid, outside get low.
    Clearly an ESTIMATE - not raw model output."""
    mask_img = Image.fromarray(binary, mode="L")
    soft = mask_img.filter(ImageFilter.GaussianBlur(radius=8))
    arr = np.array(soft).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def compute_uncertainty(prob: np.ndarray) -> dict:
    """Compute uncertainty stats from a probability map.
    Uses entropy: H = -p log p - (1-p) log (1-p), peaks at p=0.5."""
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2.0)  # normalized 0..1

    # Confident regions = where p is far from 0.5
    confident_mask = (prob >= 0.7) | (prob <= 0.3)
    uncertain_mask = (prob > 0.3) & (prob < 0.7)

    # Mean confidence over predicted tumor region (p > 0.5)
    tumor_region = prob > 0.5
    tumor_conf = float(prob[tumor_region].mean()) if tumor_region.any() else 0.0

    return {
        "mean_entropy": float(entropy.mean()),
        "max_entropy": float(entropy.max()),
        "uncertain_pct": float(uncertain_mask.mean() * 100),
        "confident_pct": float(confident_mask.mean() * 100),
        "tumor_confidence": tumor_conf,
        "entropy_map": entropy,
    }


def render_confidence_map(original: Image.Image, prob: np.ndarray, opacity: float = 0.55) -> Image.Image:
    """Color-code probability: blue (low) -> green (mid) -> red (high)."""
    h, w = prob.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Blue->Green->Red colormap (jet-like, simplified)
    rgba[..., 0] = np.clip((prob - 0.5) * 2 * 255, 0, 255).astype(np.uint8)        # red rises after 0.5
    rgba[..., 1] = np.clip((1 - np.abs(prob - 0.5) * 2) * 255, 0, 255).astype(np.uint8)  # green peaks at 0.5
    rgba[..., 2] = np.clip((0.5 - prob) * 2 * 255, 0, 255).astype(np.uint8)        # blue high before 0.5
    rgba[..., 3] = (prob * 255 * opacity).clip(0, 255).astype(np.uint8)
    color_layer = Image.fromarray(rgba, mode="RGBA")
    base = original.convert("RGBA")
    return Image.alpha_composite(base, color_layer).convert("RGB")


def render_uncertainty_map(original: Image.Image, entropy: np.ndarray) -> Image.Image:
    """Highlight uncertain pixels (high entropy) in magenta."""
    h, w = entropy.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 236
    rgba[..., 1] = 72
    rgba[..., 2] = 153
    rgba[..., 3] = (entropy * 255 * 0.7).clip(0, 255).astype(np.uint8)
    layer = Image.fromarray(rgba, mode="RGBA")
    base = original.convert("RGBA")
    return Image.alpha_composite(base, layer).convert("RGB")


def make_heatmap(original: Image.Image, binary: np.ndarray) -> Image.Image:
    """Create a heatmap-style visualization where intensity falls off from tumor center."""
    h, w = binary.shape
    if (binary > 0).sum() == 0:
        return original.convert("RGB")

    # Distance transform approximation via successive blurring
    mask_img = Image.fromarray(binary, mode="L")
    soft = mask_img.filter(ImageFilter.GaussianBlur(radius=20))
    soft_arr = np.array(soft).astype(np.float32) / 255.0

    # Build heatmap RGBA: blue -> cyan -> yellow -> red
    heat = np.zeros((h, w, 4), dtype=np.uint8)
    # Red channel: high intensity = red
    heat[..., 0] = np.clip(soft_arr * 255 * 1.4, 0, 255).astype(np.uint8)
    # Green channel: mid = yellow/green
    heat[..., 1] = np.clip(np.sin(soft_arr * np.pi) * 200, 0, 255).astype(np.uint8)
    # Blue channel: low end
    heat[..., 2] = np.clip((1 - soft_arr) * 100 * (soft_arr > 0.05), 0, 255).astype(np.uint8)
    # Alpha
    heat[..., 3] = (soft_arr * 200).astype(np.uint8)

    heat_layer = Image.fromarray(heat, mode="RGBA")
    base = original.convert("RGBA")
    out = Image.alpha_composite(base, heat_layer)
    return out.convert("RGB")


def make_zoom_view(original: Image.Image, bbox, padding: int = 30, target_size: int = 400) -> Optional[Image.Image]:
    """Crop a square zoomed view around the tumor bounding box."""
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    w_img, h_img = original.size
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    side = max(x1 - x0, y1 - y0) + padding * 2
    side = max(side, 100)
    half = side // 2
    left = max(0, cx - half)
    top = max(0, cy - half)
    right = min(w_img, cx + half)
    bottom = min(h_img, cy + half)
    crop = original.crop((left, top, right, bottom))
    crop = crop.resize((target_size, target_size), Image.LANCZOS)
    # Boost contrast slightly to make it feel like a magnified view
    enhancer = ImageEnhance.Contrast(crop)
    return enhancer.enhance(1.15)


def compute_mask_metrics(binary: np.ndarray) -> dict:
    h, w = binary.shape
    total = h * w
    tumor_pixels = int((binary > 0).sum())
    ratio = (tumor_pixels / total) * 100 if total else 0.0

    bbox = None
    centroid = None
    if tumor_pixels > 0:
        ys, xs = np.where(binary > 0)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        centroid = (int(xs.mean()), int(ys.mean()))
    return {
        "total_pixels": total,
        "tumor_pixels": tumor_pixels,
        "ratio_pct": ratio,
        "bbox": bbox,
        "centroid": centroid,
        "image_h": h,
        "image_w": w,
    }


def draw_bbox_on_overlay(overlay: Image.Image, bbox, centroid) -> Image.Image:
    if bbox is None:
        return overlay
    img = overlay.copy()
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = bbox
    # Cyan box
    draw.rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], outline=(34, 211, 238), width=2)
    # Corner markers (sci-fi style)
    corner_len = 12
    for (cx_, cy_) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]:
        pass
    # Centroid crosshair
    if centroid is not None:
        cx, cy = centroid
        r = 5
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(168, 85, 247), width=2)
        draw.line([cx - 10, cy, cx + 10, cy], fill=(168, 85, 247), width=1)
        draw.line([cx, cy - 10, cx, cy + 10], fill=(168, 85, 247), width=1)
    return img


def severity_from_ratio(ratio: float) -> Tuple[str, str, int]:
    """Return (label, badge_class, score 0-100)."""
    if ratio <= 0.0:
        return "No Tumor Detected", "badge-success", 0
    if ratio < 1.0:
        return "Low Risk", "badge-info", min(int(ratio * 25), 25)
    if ratio < 5.0:
        return "Moderate Risk", "badge-warning", 25 + int((ratio - 1) * 12)
    return "High Risk", "badge-danger", min(75 + int((ratio - 5) * 4), 100)


def render_risk_gauge(score: int, label: str) -> str:
    """SVG circular risk gauge (0-100)."""
    radius = 75
    circumference = 2 * 3.14159 * radius
    progress = (score / 100) * circumference
    if score == 0:
        color = "#10b981"
    elif score < 30:
        color = "#22d3ee"
    elif score < 60:
        color = "#f59e0b"
    else:
        color = "#ef4444"

    return f"""
    <div class="gauge-wrap">
        <svg width="180" height="180" viewBox="0 0 180 180">
            <defs>
                <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="{color}" stop-opacity="0.9"/>
                    <stop offset="100%" stop-color="{color}" stop-opacity="0.5"/>
                </linearGradient>
                <filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
            </defs>
            <circle cx="90" cy="90" r="{radius}" stroke="rgba(34,211,238,0.1)" stroke-width="10" fill="none"/>
            <circle cx="90" cy="90" r="{radius}" stroke="url(#gaugeGrad)" stroke-width="10" fill="none"
                stroke-dasharray="{progress} {circumference}"
                stroke-linecap="round"
                transform="rotate(-90 90 90)"
                filter="url(#glow)"/>
            <text x="90" y="86" text-anchor="middle" fill="#f1f5f9" font-size="34" font-weight="800" font-family="Inter">{score}</text>
            <text x="90" y="108" text-anchor="middle" fill="#94a3b8" font-size="11" font-weight="600" letter-spacing="2">RISK SCORE</text>
        </svg>
        <div style="margin-top:0.4rem; color:{color}; font-weight:600; font-size:0.95rem;">{label}</div>
    </div>
    """


def build_clinical_report(filename: str, metrics: dict, severity_label: str,
                          score: int, elapsed: float) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bbox = metrics["bbox"]
    centroid = metrics["centroid"]
    bbox_str = f"({bbox[0]}, {bbox[1]}) -> ({bbox[2]}, {bbox[3]})" if bbox else "N/A"
    centroid_str = f"({centroid[0]}, {centroid[1]})" if centroid else "N/A"

    return f"""============================================================
              QBIT CLINICAL ANALYSIS REPORT
============================================================
Generated:        {now}
Analysis Engine:  U-Net Deep Learning Segmentation
Team:             Qbit AI Research
------------------------------------------------------------

PATIENT SCAN
  Filename:       {filename}
  Resolution:     {metrics['image_w']} x {metrics['image_h']} px
  Total Pixels:   {metrics['total_pixels']:,}

SEGMENTATION RESULTS
  Tumor Detected: {"YES" if metrics['tumor_pixels'] > 0 else "NO"}
  Tumor Pixels:   {metrics['tumor_pixels']:,}
  Tumor Area:     {metrics['ratio_pct']:.3f} %
  Bounding Box:   {bbox_str}
  Centroid:       {centroid_str}

RISK ASSESSMENT
  Risk Score:     {score} / 100
  Classification: {severity_label}

PROCESSING
  Inference Time: {elapsed:.2f} seconds

------------------------------------------------------------
DISCLAIMER
This report was generated by an AI-assisted research tool.
It is NOT a medical diagnosis. All findings must be reviewed
and confirmed by a qualified radiologist or medical doctor.
============================================================

  Powered by Qbit | qbit.ai
"""


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ============================================================
# Comparison slider (Before / After) - HTML component
# ============================================================
def render_compare_slider(before_img: Image.Image, after_img: Image.Image, height: int = 520) -> None:
    """Interactive drag-to-compare slider between two images."""
    def to_b64(im: Image.Image) -> str:
        buf = BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    b64_before = to_b64(before_img)
    b64_after = to_b64(after_img)

    html = f"""
    <div id="qbit-compare" style="position:relative; width:100%; max-width:720px; margin:0 auto;
            aspect-ratio: {before_img.size[0]}/{before_img.size[1]};
            border-radius:14px; overflow:hidden;
            box-shadow:0 18px 50px rgba(0,0,0,0.55), 0 0 0 1px rgba(34,211,238,0.18);
            background:#05070d; user-select:none;">
        <img src="data:image/jpeg;base64,{b64_after}" alt="after"
            style="position:absolute; inset:0; width:100%; height:100%; object-fit:cover; display:block;" />
        <div id="qbit-clip"
            style="position:absolute; inset:0; width:50%; height:100%; overflow:hidden;">
            <img src="data:image/jpeg;base64,{b64_before}" alt="before"
                style="position:absolute; inset:0; width:200%; height:100%; object-fit:cover;
                       transform-origin: 0 0;" />
        </div>
        <div id="qbit-handle"
            style="position:absolute; top:0; bottom:0; left:50%; width:3px;
                   background:linear-gradient(180deg, #22d3ee, #a855f7);
                   box-shadow:0 0 24px rgba(34,211,238,0.85); cursor:ew-resize; z-index:5;">
            <div style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
                        width:42px; height:42px; border-radius:50%;
                        background:linear-gradient(135deg, #0f172a, #1e293b);
                        border:2px solid #22d3ee;
                        box-shadow:0 0 20px rgba(34,211,238,0.7);
                        display:flex; align-items:center; justify-content:center;
                        color:#22d3ee; font-weight:700; font-size:1rem;">⇆</div>
        </div>
        <div style="position:absolute; top:14px; left:14px; padding:0.3rem 0.7rem;
                    background:rgba(15,23,42,0.78); color:#22d3ee; font-size:0.72rem;
                    font-weight:700; letter-spacing:0.12em; border-radius:6px;
                    border:1px solid rgba(34,211,238,0.4);">ORIGINAL</div>
        <div style="position:absolute; top:14px; right:14px; padding:0.3rem 0.7rem;
                    background:rgba(15,23,42,0.78); color:#a855f7; font-size:0.72rem;
                    font-weight:700; letter-spacing:0.12em; border-radius:6px;
                    border:1px solid rgba(168,85,247,0.4);">QBIT AI</div>
    </div>
    <script>
    (function() {{
        const root = document.getElementById('qbit-compare');
        const clip = document.getElementById('qbit-clip');
        const handle = document.getElementById('qbit-handle');
        let dragging = false;

        function setPos(clientX) {{
            const rect = root.getBoundingClientRect();
            let x = clientX - rect.left;
            x = Math.max(0, Math.min(rect.width, x));
            const pct = (x / rect.width) * 100;
            clip.style.width = pct + '%';
            handle.style.left = pct + '%';
        }}

        root.addEventListener('mousedown', e => {{ dragging = true; setPos(e.clientX); }});
        window.addEventListener('mousemove', e => {{ if (dragging) setPos(e.clientX); }});
        window.addEventListener('mouseup', () => {{ dragging = false; }});
        root.addEventListener('touchstart', e => {{ dragging = true; setPos(e.touches[0].clientX); }});
        window.addEventListener('touchmove', e => {{ if (dragging) setPos(e.touches[0].clientX); }});
        window.addEventListener('touchend', () => {{ dragging = false; }});
    }})();
    </script>
    """
    components.html(html, height=height, scrolling=False)


# ============================================================
# PDF Report Generator
# ============================================================
def build_pdf_report(
    filename: str,
    metrics: dict,
    severity_label: str,
    score: int,
    elapsed: float,
    uncertainty: Optional[dict],
    has_real_confidence: bool,
    overlay_img: Image.Image,
    confidence_img: Optional[Image.Image],
) -> Optional[bytes]:
    """Generate a professional PDF clinical report. Returns None if reportlab not installed."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib.colors import HexColor, white
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
    except ImportError:
        return None

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    # Brand header bar
    c.setFillColor(HexColor("#0a0f1f"))
    c.rect(0, H - 3.2 * cm, W, 3.2 * cm, fill=1, stroke=0)
    c.setFillColor(HexColor("#22d3ee"))
    c.setFont("Helvetica-Bold", 24)
    c.drawString(2 * cm, H - 1.6 * cm, "Qbit")
    c.setFillColor(HexColor("#a855f7"))
    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, H - 2.2 * cm, "MEDICAL AI SUITE  |  Brain Tumor Segmentation")
    c.setFillColor(HexColor("#94a3b8"))
    c.setFont("Helvetica", 9)
    c.drawRightString(W - 2 * cm, H - 1.5 * cm, datetime.now().strftime("%B %d, %Y"))
    c.drawRightString(W - 2 * cm, H - 2.0 * cm, datetime.now().strftime("%H:%M:%S"))
    c.drawRightString(W - 2 * cm, H - 2.5 * cm, "Report ID: QBT-" + datetime.now().strftime("%Y%m%d%H%M%S"))

    # Section: Case info
    y = H - 4.5 * cm
    c.setFillColor(HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2 * cm, y, "CASE INFORMATION")
    c.setStrokeColor(HexColor("#22d3ee"))
    c.setLineWidth(1.5)
    c.line(2 * cm, y - 0.15 * cm, 6 * cm, y - 0.15 * cm)

    y -= 0.9 * cm
    c.setFont("Helvetica", 10)
    c.setFillColor(HexColor("#334155"))
    rows = [
        ("Source File", filename),
        ("Inference Time", f"{elapsed:.2f} seconds"),
        ("Model Architecture", "U-Net (Convolutional Encoder-Decoder)"),
        ("Confidence Source", "Raw model probabilities" if has_real_confidence else "Estimated from mask (fallback)"),
    ]
    for label, value in rows:
        c.setFillColor(HexColor("#64748b"))
        c.drawString(2 * cm, y, label + ":")
        c.setFillColor(HexColor("#0f172a"))
        c.drawString(6 * cm, y, str(value))
        y -= 0.55 * cm

    # Section: Findings
    y -= 0.4 * cm
    c.setFillColor(HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2 * cm, y, "SEGMENTATION FINDINGS")
    c.setStrokeColor(HexColor("#a855f7"))
    c.line(2 * cm, y - 0.15 * cm, 7.5 * cm, y - 0.15 * cm)

    y -= 0.9 * cm
    bbox = metrics.get("bbox")
    bbox_str = f"{bbox[2]-bbox[0]} x {bbox[3]-bbox[1]} px" if bbox else "N/A"
    centroid = metrics.get("centroid")
    centroid_str = f"({centroid[0]}, {centroid[1]})" if centroid else "N/A"

    findings = [
        ("Tumor Area Ratio", f"{metrics['ratio_pct']:.2f} %"),
        ("Tumor Pixel Count", f"{metrics['tumor_pixels']:,}"),
        ("Bounding Box", bbox_str),
        ("Centroid (x, y)", centroid_str),
        ("Risk Score", f"{score} / 100"),
        ("Severity Classification", severity_label),
    ]
    c.setFont("Helvetica", 10)
    for label, value in findings:
        c.setFillColor(HexColor("#64748b"))
        c.drawString(2 * cm, y, label + ":")
        c.setFillColor(HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 10)
        c.drawString(6 * cm, y, str(value))
        c.setFont("Helvetica", 10)
        y -= 0.55 * cm

    # Section: Uncertainty
    if uncertainty is not None:
        y -= 0.4 * cm
        c.setFillColor(HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 13)
        c.drawString(2 * cm, y, "MODEL UNCERTAINTY")
        c.setStrokeColor(HexColor("#ec4899"))
        c.line(2 * cm, y - 0.15 * cm, 6.5 * cm, y - 0.15 * cm)

        y -= 0.9 * cm
        unc_rows = [
            ("Mean Tumor Confidence", f"{uncertainty['tumor_confidence'] * 100:.1f} %"),
            ("Confident Pixels (p<.3 or p>.7)", f"{uncertainty['confident_pct']:.1f} %"),
            ("Uncertain Pixels (.3<p<.7)", f"{uncertainty['uncertain_pct']:.1f} %"),
            ("Mean Entropy", f"{uncertainty['mean_entropy']:.3f}"),
        ]
        c.setFont("Helvetica", 10)
        for label, value in unc_rows:
            c.setFillColor(HexColor("#64748b"))
            c.drawString(2 * cm, y, label + ":")
            c.setFillColor(HexColor("#0f172a"))
            c.setFont("Helvetica-Bold", 10)
            c.drawString(7 * cm, y, str(value))
            c.setFont("Helvetica", 10)
            y -= 0.55 * cm

    # Embed overlay image
    try:
        thumb = overlay_img.copy()
        thumb.thumbnail((520, 520))
        img_buf = BytesIO()
        thumb.convert("RGB").save(img_buf, format="JPEG", quality=88)
        img_buf.seek(0)
        img_reader = ImageReader(img_buf)
        iw, ih = thumb.size
        max_w = 8 * cm
        scale = max_w / iw
        draw_w, draw_h = iw * scale, ih * scale
        img_x = W - 2 * cm - draw_w
        img_y = H - 4.5 * cm - draw_h
        c.drawImage(img_reader, img_x, img_y, width=draw_w, height=draw_h)
        c.setFillColor(HexColor("#64748b"))
        c.setFont("Helvetica-Oblique", 8)
        c.drawCentredString(img_x + draw_w / 2, img_y - 0.4 * cm, "Tumor Overlay Visualization")
    except Exception:
        pass

    # Footer / disclaimer
    c.setFillColor(HexColor("#fef3c7"))
    c.rect(2 * cm, 1.8 * cm, W - 4 * cm, 1.6 * cm, fill=1, stroke=0)
    c.setFillColor(HexColor("#92400e"))
    c.setFont("Helvetica-Bold", 9)
    c.drawString(2.3 * cm, 2.9 * cm, "MEDICAL DISCLAIMER")
    c.setFont("Helvetica", 8)
    c.drawString(2.3 * cm, 2.45 * cm,
                 "This report is generated by an AI research tool for educational purposes only.")
    c.drawString(2.3 * cm, 2.10 * cm,
                 "It is not a substitute for professional medical diagnosis. Consult a qualified radiologist.")

    c.setFillColor(HexColor("#94a3b8"))
    c.setFont("Helvetica", 7)
    c.drawCentredString(W / 2, 1.0 * cm,
                        "Generated by Qbit Medical AI Suite  |  qbit.research  |  © " +
                        str(datetime.now().year))

    c.showPage()
    c.save()
    return buf.getvalue()


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown(
        '<div style="padding:0.5rem 0 1rem;">'
        '<div style="display:flex; align-items:center; gap:0.7rem;">'
        '<div class="qbit-logo" style="width:38px;height:38px;font-size:1.1rem;">Q</div>'
        '<div><div class="brand-name" style="font-size:1rem;">Qbit</div>'
        '<div class="brand-sub">Medical AI Suite</div></div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### System Status")
    api_exists = Path(API_SCRIPT).exists()
    if api_exists:
        st.markdown(
            '<span class="badge badge-success"><span class="pulse-dot"></span>Inference Engine Online</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="badge badge-danger">api.py Not Found</span>', unsafe_allow_html=True)

    st.markdown("### Visualization")
    overlay_opacity = st.slider("Overlay Opacity", 0.1, 0.9, 0.45, 0.05,
                                help="Strength of the tumor shading.")
    show_bbox = st.checkbox("Show Bounding Box & Centroid", value=True)
    show_contour_only = st.checkbox("Outline Only (no fill)", value=False)
    color_choice = st.selectbox("Highlight Color", ["Red", "Magenta", "Cyan", "Lime"], index=0)
    COLOR_MAP = {
        "Red":     (239, 68, 68),
        "Magenta": (236, 72, 153),
        "Cyan":    (34, 211, 238),
        "Lime":    (132, 204, 22),
    }
    highlight_color = COLOR_MAP[color_choice]

    st.markdown("### Backend")
    st.code("python api.py --file <image>", language="bash")
    st.caption("Inference is delegated to your existing `api.py` pipeline.")

    st.markdown("### About Qbit")
    st.caption(
        "Qbit is an AI-powered medical imaging suite built by the Qbit research team. "
        "This module performs U-Net based brain tumor segmentation. "
        "For research and educational use only."
    )

# ============================================================
# Brand bar (top)
# ============================================================
st.markdown(
    """
    <div class="brand-bar">
        <div class="brand-left">
            <div class="qbit-logo">Q</div>
            <div>
                <div class="brand-name">Qbit</div>
                <div class="brand-sub">Brain Tumor Segmentation Suite</div>
            </div>
        </div>
        <div style="display:flex; gap:0.5rem;">
            <span class="badge badge-info">U-Net</span>
            <span class="badge badge-purple">v2.0</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Header
# ============================================================
st.markdown("# Brain Tumor Segmentation")
st.markdown(
    '<p style="font-size:1.05rem; color:#94a3b8; margin-top:-0.5rem; max-width:840px;">'
    "Upload an MRI scan and Qbit's deep learning engine will detect, highlight and quantify tumor regions. "
    "Results include an interactive overlay, heatmap, magnified zoom view, and a downloadable clinical report."
    "</p>",
    unsafe_allow_html=True,
)

if not api_exists:
    st.error(
        f"`{API_SCRIPT}` was not found in the project root. "
        "This UI delegates inference to your `api.py` script - please add it before continuing."
    )
    st.stop()

# ============================================================
# Step 1: Upload
# ============================================================
st.markdown('<div class="section-title"><span class="section-num">1</span>Upload MRI Scan</div>',
            unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Select an MRI image to analyze",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    label_visibility="collapsed",
)

if not uploaded:
    st.markdown(
        '<div class="info-card">'
        "<strong style='color:#f1f5f9;'>Get started</strong><br>"
        "<span style='color:#94a3b8;'>Drag &amp; drop an MRI scan above, or click to browse. "
        "Supported formats: PNG, JPG, JPEG, BMP, TIFF.</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="qbit-footer">Powered by <span>Qbit AI Research</span></div>',
                unsafe_allow_html=True)
    st.stop()

input_path = TEMP_DIR / uploaded.name
with open(input_path, "wb") as f:
    f.write(uploaded.getbuffer())

try:
    img_obj = Image.open(input_path).convert("RGB")
    img_w, img_h = img_obj.size
    img_format = Image.open(input_path).format or "—"
except Exception as e:
    st.error(f"Could not read image: {e}")
    st.stop()

file_size_kb = os.path.getsize(input_path) / 1024

# ============================================================
# Step 2: Preview + metadata
# ============================================================
st.markdown('<div class="section-title"><span class="section-num">2</span>Image Preview</div>',
            unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Filename</div>'
        f'<div class="metric-value" style="font-size:0.95rem;">{uploaded.name}</div></div>',
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Dimensions</div>'
        f'<div class="metric-value">{img_w}×{img_h}</div></div>',
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">File Size</div>'
        f'<div class="metric-value">{file_size_kb:.1f} KB</div></div>',
        unsafe_allow_html=True,
    )
with m4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Format</div>'
        f'<div class="metric-value">{img_format}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")
pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
with pcol2:
    st.image(str(input_path), caption="Original MRI Scan", use_column_width=True)

# ============================================================
# Step 3: Analyze
# ============================================================
st.markdown('<div class="section-title"><span class="section-num">3</span>Run Analysis</div>',
            unsafe_allow_html=True)

acol1, acol2, acol3 = st.columns([1, 2, 1])
with acol2:
    analyze = st.button("Analyze Scan", type="primary")

if not analyze:
    st.markdown('<div class="qbit-footer">Powered by <span>Qbit AI Research</span></div>',
                unsafe_allow_html=True)
    st.stop()

with st.spinner("Qbit engine running U-Net segmentation..."):
    output_path, error, elapsed = run_inference(input_path)

if error or not output_path:
    st.error(f"Inference failed: {error}")
    st.stop()

# ============================================================
# Step 4: Build visualizations
# ============================================================
binary_mask = load_mask_as_binary(output_path, target_size=(img_w, img_h))
metrics = compute_mask_metrics(binary_mask)

severity_label, severity_badge, risk_score = severity_from_ratio(metrics["ratio_pct"])

fill_opacity = 0.0 if show_contour_only else overlay_opacity
overlay_img = make_overlay(img_obj, binary_mask, opacity=fill_opacity, color=highlight_color)
if show_bbox:
    overlay_img = draw_bbox_on_overlay(overlay_img, metrics["bbox"], metrics["centroid"])

mask_render = colorize_mask(binary_mask, color=highlight_color)
mask_on_black = Image.new("RGB", mask_render.size, (5, 8, 20))
mask_on_black.paste(mask_render, mask=mask_render.split()[-1])

heatmap_img = make_heatmap(img_obj, binary_mask)
zoom_img = make_zoom_view(img_obj, metrics["bbox"])
zoom_overlay = make_zoom_view(overlay_img, metrics["bbox"]) if metrics["bbox"] else None

# ===== Real Confidence / Uncertainty =====
prob_path = find_probability_map(input_path)
has_real_confidence = prob_path is not None
if has_real_confidence:
    prob_map = load_probability_map(prob_path, target_size=(img_w, img_h))
else:
    prob_map = estimate_confidence_from_mask(binary_mask)

uncertainty = compute_uncertainty(prob_map)
confidence_img = render_confidence_map(img_obj, prob_map, opacity=0.55)
uncertainty_img = render_uncertainty_map(img_obj, uncertainty["entropy_map"])

# ============================================================
# Step 5: Results
# ============================================================
st.markdown('<div class="section-title"><span class="section-num">4</span>Diagnostic Results</div>',
            unsafe_allow_html=True)

# Status badges row
sr1, sr2, sr3 = st.columns([2, 2, 2])
with sr1:
    st.markdown(
        f'<span class="badge badge-success"><span class="pulse-dot"></span>Analysis complete in {elapsed:.2f}s</span>',
        unsafe_allow_html=True,
    )
with sr2:
    st.markdown(
        f'<span class="badge {severity_badge}">{severity_label}</span>',
        unsafe_allow_html=True,
    )
with sr3:
    st.markdown(
        f'<span class="badge badge-info">Resolution: {img_w}×{img_h}</span>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ===== Risk Gauge + Metrics row =====
gcol, mcol = st.columns([1, 2])
with gcol:
    st.markdown(render_risk_gauge(risk_score, severity_label), unsafe_allow_html=True)

with mcol:
    ratio = metrics["ratio_pct"]
    ratio_class = "ok" if ratio == 0 else ("warn" if ratio < 5 else "danger")
    bbox = metrics["bbox"]
    bbox_str = f'{bbox[2]-bbox[0]}×{bbox[3]-bbox[1]}' if bbox else "—"
    centroid_str = f'({metrics["centroid"][0]}, {metrics["centroid"][1]})' if metrics["centroid"] else "—"

    mm1, mm2 = st.columns(2)
    with mm1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Tumor Area</div>'
            f'<div class="metric-value {ratio_class}">{ratio:.2f}%</div></div>',
            unsafe_allow_html=True,
        )
    with mm2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Tumor Pixels</div>'
            f'<div class="metric-value">{metrics["tumor_pixels"]:,}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    mm3, mm4 = st.columns(2)
    with mm3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Bounding Box</div>'
            f'<div class="metric-value purple" style="font-size:1.2rem;">{bbox_str}</div></div>',
            unsafe_allow_html=True,
        )
    with mm4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Centroid (x, y)</div>'
            f'<div class="metric-value purple" style="font-size:1.2rem;">{centroid_str}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("")

# ===== Tabbed views =====
tab_titles = ["Tumor Overlay", "Compare", "Confidence Map", "Heatmap", "Zoom View", "Side by Side", "Mask Only"]
tabs = st.tabs(tab_titles)

with tabs[0]:
    st.markdown("**Tumor highlighted on the original scan** — translucent fill + glow halo + contour outline.")
    c1, c2, c3 = st.columns([1, 4, 1])
    with c2:
        st.image(overlay_img, use_column_width=True)

with tabs[1]:
    st.markdown(
        "**Drag the slider** to compare the original scan with the Qbit AI prediction. "
        "Move the cyan handle left and right."
    )
    render_compare_slider(img_obj, overlay_img, height=560)

with tabs[2]:
    if has_real_confidence:
        st.markdown(
            '<span class="badge badge-success">Real model probabilities loaded from '
            f'<code>{prob_path.name}</code></span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="badge badge-warning" style="background:rgba(245,158,11,0.18); '
            'color:#fbbf24; border:1px solid rgba(245,158,11,0.4);">'
            'Estimated confidence (no <code>*_prob.npy</code> output found from api.py)</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Tip: Update api.py to additionally save the raw sigmoid output as "
            "`<stem>_prob.npy` to enable true model uncertainty visualization."
        )

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Confidence (probability) map** — blue=low, green=mid, red=high.")
        st.image(confidence_img, use_column_width=True)
    with cc2:
        st.markdown("**Uncertainty (entropy)** — magenta = pixels the model is unsure about.")
        st.image(uncertainty_img, use_column_width=True)

    u1, u2, u3, u4 = st.columns(4)
    with u1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Mean Tumor Confidence</div>'
            f'<div class="metric-value">{uncertainty["tumor_confidence"]*100:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with u2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Confident Pixels</div>'
            f'<div class="metric-value ok">{uncertainty["confident_pct"]:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with u3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Uncertain Pixels</div>'
            f'<div class="metric-value warn">{uncertainty["uncertain_pct"]:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with u4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Mean Entropy</div>'
            f'<div class="metric-value purple">{uncertainty["mean_entropy"]:.3f}</div></div>',
            unsafe_allow_html=True,
        )

with tabs[3]:
    st.markdown("**Heatmap visualization** — intensity falls off from the tumor center.")
    c1, c2, c3 = st.columns([1, 4, 1])
    with c2:
        st.image(heatmap_img, use_column_width=True)

with tabs[4]:
    if zoom_img is not None:
        st.markdown("**Magnified view of the tumor region.**")
        zc1, zc2 = st.columns(2)
        with zc1:
            st.markdown("**Zoomed Original**")
            st.image(zoom_img, use_column_width=True)
        with zc2:
            st.markdown("**Zoomed Overlay**")
            if zoom_overlay is not None:
                st.image(zoom_overlay, use_column_width=True)
    else:
        st.info("No tumor region detected — zoom view unavailable.")

with tabs[5]:
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Original**")
        st.image(img_obj, use_column_width=True)
    with cB:
        st.markdown("**Predicted Overlay**")
        st.image(overlay_img, use_column_width=True)

with tabs[6]:
    st.markdown("**Predicted binary segmentation mask.**")
    c1, c2, c3 = st.columns([1, 4, 1])
    with c2:
        st.image(mask_on_black, use_column_width=True)

st.markdown("")

# ===== Downloads =====
st.markdown('<div class="section-title"><span class="section-num">5</span>Downloads</div>',
            unsafe_allow_html=True)

d1, d2, d3, d4, d5 = st.columns(5)
with d1:
    st.download_button(
        "Overlay PNG",
        data=pil_to_bytes(overlay_img),
        file_name=f"{input_path.stem}_overlay.png",
        mime="image/png",
    )
with d2:
    st.download_button(
        "Confidence PNG",
        data=pil_to_bytes(confidence_img),
        file_name=f"{input_path.stem}_confidence.png",
        mime="image/png",
    )
with d3:
    st.download_button(
        "Heatmap PNG",
        data=pil_to_bytes(heatmap_img),
        file_name=f"{input_path.stem}_heatmap.png",
        mime="image/png",
    )
with d4:
    st.download_button(
        "Mask PNG",
        data=pil_to_bytes(mask_on_black),
        file_name=f"{input_path.stem}_mask.png",
        mime="image/png",
    )
with d5:
    pdf_bytes = build_pdf_report(
        filename=uploaded.name,
        metrics=metrics,
        severity_label=severity_label,
        score=risk_score,
        elapsed=elapsed,
        uncertainty=uncertainty,
        has_real_confidence=has_real_confidence,
        overlay_img=overlay_img,
        confidence_img=confidence_img,
    )
    if pdf_bytes is not None:
        st.download_button(
            "PDF Report",
            data=pdf_bytes,
            file_name=f"{input_path.stem}_qbit_report.pdf",
            mime="application/pdf",
        )
    else:
        # Fallback: TXT report if reportlab is not installed
        report_text = build_clinical_report(
            filename=uploaded.name,
            metrics=metrics,
            severity_label=severity_label,
            score=risk_score,
            elapsed=elapsed,
        )
        st.download_button(
            "TXT Report",
            data=report_text,
            file_name=f"{input_path.stem}_qbit_report.txt",
            mime="text/plain",
            help="Install `reportlab` (pip install reportlab) to enable PDF export.",
        )

# ===== Disclaimer =====
st.markdown(
    '<div class="info-card" style="margin-top:1.5rem; border-color: rgba(245,158,11,0.25);">'
    "<strong style='color:#f59e0b;'>Medical Disclaimer</strong><br>"
    "<span style='color:#cbd5e1;'>This tool is intended for research and educational purposes only. "
    "It is not a substitute for professional medical diagnosis. Always consult a qualified radiologist "
    "or medical professional for clinical decisions.</span>"
    "</div>",
    unsafe_allow_html=True,
)

# ===== Footer =====
st.markdown(
    '<div class="qbit-footer">'
    'Powered by <span>Qbit AI Research</span> &middot; Built with Streamlit, NumPy, and PIL'
    '</div>',
    unsafe_allow_html=True,
)
