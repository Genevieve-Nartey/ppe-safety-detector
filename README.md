# 🦺 Construction Site PPE Safety Detector

> Real-time Personal Protective Equipment (PPE) detection system using YOLOv8 — flags workers missing hard hats, safety vests, gloves, and boots on construction sites.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)
![Gradio](https://img.shields.io/badge/Demo-Gradio-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Problem

Construction sites are one of the most dangerous workplaces globally. A significant portion of injuries are caused by workers not wearing required PPE. Manual inspection is slow, inconsistent, and expensive.

This project automates PPE compliance monitoring using a fine-tuned YOLOv8 object detection model — capable of running in real-time on standard hardware.

---

## 🚀 Live Demo

👉 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/ppe-safety-detector)**

> *Upload a construction site image → model detects PPE and flags violations in real time*

---

## 📊 Results

All models trained for 100 epochs on the same dataset. Evaluated on validation set.

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Inference (ms) | Params |
|-------|---------|--------------|-----------|--------|----------------|--------|
| YOLOv8n | 38.1% | 27.0% | 77.8% | 30.2% | 2.5ms | 3.0M |
| **YOLOv8s** | **39.2%** | **25.3%** | **70.5%** | **35.3%** | **4.8ms** | **11.1M** |
| YOLOv8m | 39.1% | 23.0% | 81.6% | 32.9% | 11.1ms | 25.8M |

**✅ Final model: YOLOv8s** — best mAP@0.5 with 4.8ms inference. The nano model had the highest mAP@0.5:0.95 (27.0%) suggesting better localisation, but small edges it out overall for balanced real-time use.

### Per-Class Performance (YOLOv8s, best.pt)

| Class | mAP@0.5 | Notes |
|-------|---------|-------|
| safety-vest | 99.5% | Easiest — high contrast, large area |
| safety-boots | 83.2% | Strong performance |
| gloves | 68.3% | Small objects, harder to detect |
| no-safety-vest | 0% | Insufficient training examples |
| no-safety-boots | ~1% | Rare in dataset |
| person | ~4.4% | Needs more diverse training data |
| machinery | 17.9% | Limited examples |

> **Note:** Violation classes (no-\*) have near-zero recall due to dataset imbalance — only ~200 effective training images after filtering corrupt labels. This is the primary area for improvement.

---

## 🧪 Experiments

### Experiment 1 — Model Scale
Trained nano/small/medium with identical hyperparameters. Surprisingly, nano achieved the highest mAP@0.5:0.95 (27.0%), likely due to better generalisation on a small dataset. Larger models began to overfit.

### Experiment 2 — Augmentation
Mosaic augmentation (combining 4 images per sample) was applied throughout. This is especially effective for construction scenes where multiple workers appear in a single frame. HSV shifts (hue ±1.5%, saturation ±70%, value ±40%) help with variable site lighting conditions.

### Experiment 3 — Transfer Learning
All models initialised from COCO pretrained weights. The model already understands people, objects, and scenes — fine-tuning teaches it PPE-specific patterns. Training converged in ~80 epochs rather than the ~200 needed from random initialisation.

### Experiment 4 — Dataset Quality
102 of 307 training images (33%) had corrupt labels (class IDs exceeding nc=10). After filtering, only ~205 images were used for training. Fixing dataset quality is the single highest-leverage improvement available.

---

## 🏗️ Architecture

```
Input Image (640×640)
        │
   YOLOv8s Backbone (CSPDarknet)
        │
   PANet Feature Pyramid Neck
        │
   Detection Head (3 scales: 80×80, 40×40, 20×20)
        │
   Post-processing (NMS, confidence filtering)
        │
   Violation Logic Layer
        │
   Output: Bounding Boxes + Violation Alerts
```

**Key design decisions:**
- **Dual-class strategy**: Both presence (`hardhat`) and absence (`no-hardhat`) are explicit classes, avoiding post-hoc logic
- **Mosaic augmentation**: Simulates crowded multi-worker scenes common on construction sites
- **COCO pretraining**: Transfers general object knowledge, dramatically reducing data needed

---

## 📁 Project Structure

```
ppe-safety-detector/
│
├── data/                          # Dataset (not committed — download via Roboflow)
│   ├── train/
│   ├── valid/
│   └── test/
│
├── app.py                         # FastAPI REST API
├── demo.py                        # Gradio interactive demo (deployed to HF Spaces)
├── construction_safety.yaml       # Dataset config
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Usage

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/ppe-safety-detector.git
cd ppe-safety-detector
pip install -r requirements.txt
```

### 2. Download Dataset

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("roboflow-universe-projects").project("construction-site-safety")
dataset = project.version(1).download("yolov8", location="./data")
```

### 3. Run the API

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

**Example response:**
```json
{
  "num_detections": 2,
  "detections": [
    {"label": "safety-vest", "confidence": 0.96, "is_violation": false},
    {"label": "no-hardhat",  "confidence": 0.87, "is_violation": true}
  ],
  "violations": ["⚠️ Worker missing hard hat"],
  "safe": false,
  "inference_ms": 4.8
}
```

### 4. Launch Gradio Demo

```bash
python demo.py
# Opens at http://localhost:7860
```

---

## 🗺️ Future Work

- [ ] **Fix dataset quality** — remove/relabel the 102 corrupt training images
- [ ] **Collect more violation examples** — no-\* classes have near-zero recall due to underrepresentation
- [ ] **Video inference** — process RTSP streams from IP cameras
- [ ] **Multi-camera tracking** — persistent worker IDs across frames using ByteTrack
- [ ] **Edge deployment** — export to ONNX / TensorRT for Jetson Nano
- [ ] **Severity scoring** — weight violations by risk (missing hard hat > missing gloves)

---

## 📦 Dataset

**Source:** [Roboflow Construction Site Safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)
**Effective training images:** ~205 (after filtering corrupt labels)
**Classes:** 10 (5 PPE types × present/absent + person + machinery)
**Augmentations:** mosaic, horizontal flip, HSV shift

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Model training & inference |
| [FastAPI](https://fastapi.tiangolo.com/) | REST API |
| [Gradio](https://gradio.app/) | Interactive demo |
| [Roboflow](https://roboflow.com/) | Dataset management |
| [Kaggle Notebooks](https://kaggle.com) | GPU training (T4 x2) |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋 About

Built as part of a Computer Vision portfolio. If you're working on similar problems in construction safety, autonomous inspection, or industrial AI — feel free to reach out.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/YOUR_USERNAME)
[![Hugging Face](https://img.shields.io/badge/🤗-Demo-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/ppe-safety-detector)
