# YOLOv5 Person Detection & Multi-Object Tracking (SORT → DeepSORT)

This project demonstrates an end-to-end **person detection and tracking pipeline** using **YOLOv5**, **SORT**, and **DeepSORT**.  
The goal is to move from raw object detection to **robust multi-object tracking** in real-world video scenarios.

---

## 🚀 Project Overview

The pipeline consists of three main stages:

1. **Person Detection (YOLOv5)**
2. **Tracking with SORT**
3. **Tracking with DeepSORT (Appearance-based Re-Identification)**

Each stage is evaluated visually and comparatively to highlight the strengths and limitations of different tracking approaches.

---

## 📁 Dataset

- **Merged dataset** created from:
  - Roboflow person datasets
  - Kaggle public person datasets
- Unified into a **single person-only dataset**
- Clean label structure and consistent annotation format

### Why a merged dataset?
- Improves **data diversity** (indoor, outdoor, different lighting conditions)
- Reduces overfitting to a single data source
- Leads to more stable detection performance in real-world videos

---

## 🧠 Model Training (YOLOv5)

- Architecture: **YOLOv5**
- Image size: **640 × 640**
- Training epochs: **100**
- Single class: `person`
- Optimized confidence threshold: **0.50**

### Why YOLOv5?
- Excellent speed–accuracy balance
- Mature ecosystem and tooling
- Easy integration with tracking pipelines
- Well-suited for real-time applications

---

## 📊 Evaluation & Metrics

- **mAP ≈ 0.49**
- Precision–Recall and F1 curves analyzed
- Confidence threshold tuned to reduce false positives
- Priority given to **stable tracking quality**, not only raw recall

---

## 🎯 Tracking Approaches

### 1️⃣ SORT (Simple Online and Realtime Tracking)

- Uses **IoU + Kalman Filter**
- Very fast and lightweight
- Limitations:
  - Frequent **ID switches** in crowded scenes
  - Weak performance under **occlusion**

### 2️⃣ DeepSORT (Improved Tracking)

- Extends SORT with **appearance embeddings (Re-ID)**
- Uses visual features to re-identify people after occlusion
- Key advantages:
  - Significantly fewer ID switches
  - Better identity preservation
  - More realistic real-world behavior

---

## 🧩 Key Concepts Explained

- **ID Switch**:  
  When the same person is assigned different IDs across frames.

- **Occlusion**:  
  When a person is temporarily hidden by objects or other people.

DeepSORT reduces ID switches by using appearance-based matching, allowing the tracker to recover identities after occlusions.

---

## 🎥 Demo Videos

The project includes three comparison videos:

- `detection.mp4` — YOLOv5 person detection only
- `sort.mp4` — Detection + SORT tracking
- `deepsort_conf050.mp4` — Detection + DeepSORT tracking (final result)

These videos clearly show the improvement from detection to advanced tracking.

---

## ⚙️ Implementation Details

- Confidence filtering applied before tracking (`conf ≥ 0.50`)
- DeepSORT parameters tuned for stability:
  - Reduced false positives
  - Tighter appearance matching
- Frame-accurate alignment between detection outputs and video frames

---

## 📌 Limitations & Future Work

- Bicycle and vehicle tracking not included (person-only model)
- Extreme occlusions may still cause ID loss
- Future improvements:
  - Multi-class training (person + bicycle)
  - YOLOv8 comparison
  - ByteTrack integration
  - Real-time webcam inference

---

## 🧑‍💻 Author

**Burak Kaygusuzoğlu**  
Computer Engineering Student  
Focus areas:  
Computer Vision · Deep Learning · Object Detection · Multi-Object Tracking

---

## ⭐ Conclusion

This project demonstrates a complete and realistic computer vision workflow:
from dataset preparation and model training to advanced multi-object tracking.

It highlights why **tracking quality** depends not only on detection accuracy,
but also on **identity consistency and appearance modeling**.

---
Pretrained weights are not included in this repository.

You can download or train your own YOLOv5 weights using the provided scripts.
