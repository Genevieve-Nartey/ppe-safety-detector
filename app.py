"""
app.py — PPE Safety Detector REST API
======================================
Usage:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Requirements:
    pip install ultralytics fastapi uvicorn python-multipart pillow

Endpoints:
    POST /detect        — Upload image, get detections + violations
    GET  /health        — Health check
    GET  /docs          — Auto-generated Swagger UI
"""

import io
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
from pydantic import BaseModel


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH   = "ppe_detector/yolov8_medium_v1/weights/best.pt"
FALLBACK     = "yolov8m.pt"          # used if best.pt not found yet
CONF_DEFAULT = 0.45
IOU_DEFAULT  = 0.45
IMG_SIZE     = 640

VIOLATION_CLASSES = {1: "no-hardhat", 3: "no-safety-vest", 5: "no-gloves", 7: "no-safety-boots"}
VIOLATION_IDS     = set(VIOLATION_CLASSES.keys())

LABEL_NAMES = {
    0: "hardhat",       1: "no-hardhat",
    2: "safety-vest",   3: "no-safety-vest",
    4: "gloves",        5: "no-gloves",
    6: "safety-boots",  7: "no-safety-boots",
    8: "person",        9: "machinery",
}

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PPE Safety Detector API",
    description="Real-time Personal Protective Equipment detection using YOLOv8",
    version="1.0.0",
)

# Load model once at startup
_weights = MODEL_PATH if Path(MODEL_PATH).exists() else FALLBACK
print(f"🔍 Loading model from: {_weights}")
model = YOLO(_weights)
print("✅ Model loaded.")


# ── Schemas ───────────────────────────────────────────────────────────────────

class Detection(BaseModel):
    label: str
    class_id: int
    confidence: float
    bbox: list[float]          # [x1, y1, x2, y2] normalised 0–1
    is_violation: bool

class DetectionResponse(BaseModel):
    image_width: int
    image_height: int
    num_detections: int
    detections: list[Detection]
    violations: list[str]
    safe: bool
    inference_ms: float


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_image(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return img

def run_inference(img: Image.Image, conf: float, iou: float):
    t0 = time.perf_counter()
    results = model(img, conf=conf, iou=iou, imgsz=IMG_SIZE, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return results, elapsed_ms

def parse_results(results, img_w: int, img_h: int) -> tuple[list[Detection], list[str]]:
    detections = []
    violations = []

    if results.boxes is None or len(results.boxes) == 0:
        return detections, violations

    for box in results.boxes:
        cls_id  = int(box.cls.item())
        conf    = round(float(box.conf.item()), 4)
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Normalise bbox to 0–1
        bbox_norm = [
            round(x1 / img_w, 4), round(y1 / img_h, 4),
            round(x2 / img_w, 4), round(y2 / img_h, 4),
        ]

        is_violation = cls_id in VIOLATION_IDS
        label        = LABEL_NAMES.get(cls_id, f"class_{cls_id}")

        if is_violation:
            ppe_name = label.replace("no-", "").replace("-", " ")
            violations.append(f"⚠️  Worker missing {ppe_name}")

        detections.append(Detection(
            label=label,
            class_id=cls_id,
            confidence=conf,
            bbox=bbox_norm,
            is_violation=is_violation,
        ))

    return detections, violations


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": _weights}


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(..., description="Image file (JPEG/PNG)"),
    conf: float      = Query(CONF_DEFAULT, ge=0.1, le=1.0,
                             description="Confidence threshold"),
    iou:  float      = Query(IOU_DEFAULT,  ge=0.1, le=1.0,
                             description="IoU threshold for NMS"),
):
    """
    Upload an image and receive:
    - All detected objects with bounding boxes
    - List of PPE violations found
    - Whether the scene is safe (no violations)
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="File must be an image.")

    raw = await file.read()
    img = load_image(raw)
    w, h = img.size

    results, ms = run_inference(img, conf, iou)
    detections, violations = parse_results(results, w, h)

    return DetectionResponse(
        image_width=w,
        image_height=h,
        num_detections=len(detections),
        detections=detections,
        violations=list(set(violations)),        # deduplicate
        safe=len(violations) == 0,
        inference_ms=round(ms, 2),
    )


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
