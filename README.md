# VisionIQ - AI-Powered CCTV Suspicious Activity Recognition

VisionIQ is an AI-powered computer vision system designed for real-time video surveillance and automated detection of anomalous and suspicious events from CCTV video streams.

The core AI component utilizes spatiotemporal deep learning models (3D decomposed convolutions via **R(2+1)D-18** pretrained on Kinetics-400 and an edge-compatible **MobileNetV3 + Bi-GRU** fallback) fine-tuned for surveillance anomaly recognition.

---

## 🎯 Target Activity Categories

VisionIQ classifies video activity into 7 discrete categories:
1. **Normal / Non-suspicious Activity**
2. **Fighting**
3. **Robbery**
4. **Shoplifting**
5. **Burglary**
6. **Road Accidents**
7. **Vandalism**

---

## 🏗️ System Architecture

* **Input:** CCTV surveillance videos (MP4, H.264).
* **Frame Sampling:** Uniform temporal extraction of $T = 16$ frames per clip, resized to $112 \times 112$ with Kinetics-400 channel normalization.
* **Backbone:** **R(2+1)D-18** with decomposed spatial $(1 \times 3 \times 3)$ and temporal $(3 \times 1 \times 1)$ convolutions.
* **Optimization:** PyTorch Automatic Mixed Precision (FP16 AMP) with early-layer feature freezing (`--freeze_early`), optimized for 4 GB GPUs (NVIDIA GeForce GTX 1650).
* **Zero-Leakage Guarantee:** Train, validation, and test sets are partitioned strictly by source camera recording (`parent_video_id`) to ensure no background scene memorization or data leakage.

---

## 📁 Repository Structure

```
VisionIQ/
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excludes datasets, videos, and large checkpoints
├── splits/                     # Leakage-free dataset split metadata
│   ├── train.csv               # 3,592 training clips (205 source recordings)
│   ├── val.csv                 # 788 validation clips (44 source recordings)
│   └── test.csv                # 797 test clips (44 source recordings)
├── src/
│   ├── __init__.py             # Module initialization
│   ├── config.py               # Hyperparameters, paths, and device setup
│   ├── dataset.py              # Zero-leakage splits & on-the-fly PyTorch video loader
│   ├── model.py                # Pretrained R(2+1)D-18 and MobileNetV3-GRU architectures
│   ├── train.py                # FP16 AMP training loop with live tqdm progress bars
│   ├── evaluate.py             # Precision, Recall, F1, and Confusion Matrix generator
│   └── inference.py            # Sliding-window anomaly detector for unseen CCTV videos
├── tests/
│   └── verify_pipeline.py      # Automated 5-stage pipeline unit verification suite
├── checkpoints/                # Model checkpoints, epoch logs, and loss curves
└── outputs/                    # Confusion matrix plots and evaluation metrics
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Verify Pipeline Integrity
Run the automated verification suite (tests zero-leakage splits, DataLoader shapes, GPU forward pass, VRAM safety, and convergence):
```bash
python tests/verify_pipeline.py
```

### 3. Training
Fine-tune the pretrained video action recognition model:
```bash
python -m src.train --epochs 10 --batch_size 4 --lr 1e-4
```
* **Best checkpoint:** Saved automatically to `checkpoints/best_model.pth` based on Validation Macro F1.
* **Logs & Curves:** Metrics logged to `checkpoints/training_log.csv` and plotted to `checkpoints/loss_curve.png`.

### 4. Evaluation
Evaluate model performance on the held-out test split:
```bash
python -m src.evaluate --batch_size 8
```
Outputs classification metrics (Accuracy, Precision, Recall, Macro F1) and saves a normalized confusion matrix to `outputs/confusion_matrix.png`.

### 5. Inference on Unseen CCTV Footage
Run sliding-window activity detection on any continuous CCTV video:
```bash
python -m src.inference --video "path/to/cctv_video.mp4" --threshold 0.55 --window_sec 2.0 --stride_sec 1.0
```
Outputs timestamped candidate anomaly events:
```
[01] [00:14.00 -> 00:22.00] ALERT: FIGHTING (Peak Conf: 87.4%, Avg Conf: 81.2%)
```

---

## 💻 Hardware Compatibility
* **Tested GPU:** NVIDIA GeForce GTX 1650 (4 GB VRAM)
* **Peak Training VRAM:** ~1.22 GB (safely below 4 GB buffer)
* **Host RAM:** 8 GB DDR4
* **Operating System:** Windows 10 / 11 / Linux
