"""
demo.py — PPE Safety Detector — Gradio Demo
=============================================
Deploy to Hugging Face Spaces in 3 steps:
  1. Push this file + best.pt to a new HF Space (Docker or Gradio SDK)
  2. Add ultralytics + gradio to requirements.txt
  3. Set model path below

Local usage:
    pip install gradio ultralytics pillow
    python demo.py
"""

from pathlib import Path
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = "ppe_detector/yolov8_medium_v1/weights/best.pt"
FALLBACK   = "yolov8m.pt"
CONF       = 0.45

LABEL_NAMES = {
    0: "hardhat",       1: "no-hardhat",
    2: "safety-vest",   3: "no-safety-vest",
    4: "gloves",        5: "no-gloves",
    6: "safety-boots",  7: "no-safety-boots",
    8: "person",        9: "machinery",
}

# Colour palette: green = safe, red = violation, blue = neutral
COLOURS = {
    0: "#00C851",   # hardhat          → green
    1: "#FF4444",   # no-hardhat       → red
    2: "#00C851",   # safety-vest      → green
    3: "#FF4444",   # no-safety-vest   → red
    4: "#00C851",   # gloves           → green
    5: "#FF4444",   # no-gloves        → red
    6: "#00C851",   # safety-boots     → green
    7: "#FF4444",   # no-safety-boots  → red
    8: "#33B5E5",   # person           → blue
    9: "#AA66CC",   # machinery        → purple
}

VIOLATION_IDS = {1, 3, 5, 7}


# ── Model ─────────────────────────────────────────────────────────────────────

weights = MODEL_PATH if Path(MODEL_PATH).exists() else FALLBACK
print(f"Loading model: {weights}")
model = YOLO(weights)


# ── Drawing ───────────────────────────────────────────────────────────────────

def hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def draw_boxes(img: Image.Image, results) -> Image.Image:
    """Draw bounding boxes and labels on a PIL image."""
    draw = ImageDraw.Draw(img, "RGBA")
    font_size = max(14, img.width // 60)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    if results.boxes is None:
        return img

    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf   = float(box.conf.item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        colour     = COLOURS.get(cls_id, "#FFFFFF")
        rgb        = hex_to_rgb(colour)
        fill_alpha = (*rgb, 40)

        # Box fill (semi-transparent)
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
        draw.rectangle([x1, y1, x2, y2], fill=fill_alpha)

        # Label background
        label = f"{LABEL_NAMES.get(cls_id, cls_id)} {conf:.0%}"
        bbox_text = draw.textbbox((x1, y1 - font_size - 4), label, font=font)
        draw.rectangle(bbox_text, fill=colour)
        draw.text((x1, y1 - font_size - 4), label, fill="white", font=font)

    return img


# ── Core Inference Function ───────────────────────────────────────────────────

def predict(image: np.ndarray, confidence: float) -> tuple[Image.Image, str]:
    """
    Run YOLOv8 inference on the uploaded image.
    Returns annotated image + markdown report.
    """
    if image is None:
        return None, "❌ Please upload an image."

    img_pil = Image.fromarray(image).convert("RGB")
    results = model(img_pil, conf=confidence, verbose=False)[0]

    # Annotate
    annotated = draw_boxes(img_pil.copy(), results)

    # Build report
    detections = []
    violations = []

    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf   = float(box.conf.item())
            label  = LABEL_NAMES.get(cls_id, f"class_{cls_id}")
            detections.append(f"- **{label}** ({conf:.0%})")
            if cls_id in VIOLATION_IDS:
                ppe = label.replace("no-", "").replace("-", " ")
                violations.append(f"⚠️ Worker missing **{ppe}**")

    # Status
    if not detections:
        status = "🔍 No objects detected. Try lowering the confidence threshold."
    elif not violations:
        status = "✅ **Scene is SAFE** — all detected workers wearing required PPE."
    else:
        status = f"🚨 **{len(set(violations))} VIOLATION(S) DETECTED**"

    report = f"""## Detection Report

**Status:** {status}

### Detections ({len(detections)} total)
{chr(10).join(detections) if detections else "_None_"}

### Violations
{chr(10).join(set(violations)) if violations else "✅ None"}
"""

    return annotated, report


# ── Example Images ────────────────────────────────────────────────────────────

EXAMPLES = [
    ["examples/construction_safe.jpg",    0.45],
    ["examples/construction_unsafe.jpg",  0.45],
]

# ── Gradio Interface ──────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), title="PPE Safety Detector") as demo:

    gr.Markdown("""
    # 🦺 Construction Site PPE Safety Detector
    Upload a construction site image to automatically detect Personal Protective Equipment (PPE) violations.
    Detects: **hard hats, safety vests, gloves, safety boots, and people**.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Image", type="numpy")
            confidence  = gr.Slider(0.1, 0.9, value=0.45, step=0.05,
                                    label="Confidence Threshold",
                                    info="Lower = more detections, higher = fewer but more certain")
            detect_btn  = gr.Button("🔍 Detect", variant="primary")

        with gr.Column(scale=1):
            output_image  = gr.Image(label="Annotated Result")
            output_report = gr.Markdown(label="Report")

    detect_btn.click(
        fn=predict,
        inputs=[input_image, confidence],
        outputs=[output_image, output_report],
    )

    # Also run on image upload (no button press needed)
    input_image.change(
        fn=predict,
        inputs=[input_image, confidence],
        outputs=[output_image, output_report],
    )

    gr.Markdown("""
    ---
    **Model:** YOLOv8m fine-tuned on [Construction Safety Dataset](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)
    **Classes:** hardhat, no-hardhat, safety-vest, no-safety-vest, gloves, no-gloves, safety-boots, no-safety-boots, person, machinery
    """)


if __name__ == "__main__":
    demo.launch(share=True)    # share=True generates a public URL
