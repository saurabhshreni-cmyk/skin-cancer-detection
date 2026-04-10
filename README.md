
# Skin Cancer Detection using Segmentation + Classification

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![Task](https://img.shields.io/badge/Task-Medical%20Imaging-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end deep learning pipeline for skin lesion analysis that combines lesion **segmentation** and **classification** for improved diagnostic performance.

## Overview

This project follows a two-stage medical imaging workflow:

`Image -> Segmentation (U-Net) -> ROI Extraction -> Classification (EfficientNet-B0) -> Prediction + Grad-CAM`

- Stage 1: Train U-Net on ISIC 2018 to segment lesion masks.
- Stage 2: Use predicted masks to extract lesion-focused ROI crops on ISIC 2020.
- Stage 3: Train EfficientNet-B0 classifier on ROI crops (and compare with full-image baseline).
- Stage 4: Evaluate with ROC/PR curves, confusion matrix, and Grad-CAM visual explanations.

## Key Features

- U-Net segmentation model implemented from scratch.
- EfficientNet-B0 transfer learning for binary classification.
- Segmentation-guided ROI extraction (largest contour + margin, with fallback).
- Robust evaluation with:
  - ROC-AUC
  - Precision-Recall analysis
  - Confusion Matrix
  - Accuracy / Precision / Recall / F1
- Explainability via Grad-CAM overlays.
- Reproducible setup using fixed random seed and centralized config.

## Dataset Description

## ISIC 2018 (Segmentation)
- Input: dermoscopic images.
- Target: binary lesion masks (`*_segmentation.png`).
- Used for training the U-Net segmentation module.

## ISIC 2020 (Classification)
- Input: dermoscopic images.
- Target: binary label from `target` column (`0` benign, `1` malignant).
- Image identifier column: `isic_id`.
- Used for classifier training and ROI-vs-full-image comparison.
## 📊 Results

| Metric | ROI Model | Full Image Model |
|--------|----------|-----------------|
| ROC-AUC | 0.8746 | **0.8793** |
| Dice Score (Segmentation) | 0.85 | — |

### ⚠️ Note on Interpretation

Although segmentation-based ROI extraction was expected to improve performance, the full-image model slightly outperforms the ROI-based model in this setup.

This may be due to:
- Loss of contextual information during ROI cropping
- Imperfect segmentation masks
- Dataset imbalance affecting classification behavior

Despite this, Grad-CAM visualizations show improved focus on lesion regions in the ROI-based model.

### ⚠️ Note on Metrics

Although the model achieves a strong ROC-AUC (~0.88), precision is relatively low due to class imbalance in the dataset.

- ROC-AUC measures ranking ability (threshold-independent)
- Precision depends on classification threshold (default = 0.5)

👉 On imbalanced data, a fixed threshold can lead to poor precision despite good ranking performance.

This indicates that the model distinguishes classes well but requires better threshold calibration for practical use.

**Future improvements:**
- Threshold tuning
- Focal loss / class weighting
- Better handling of class imbalance

### 🔍 Confusion Matrix Insight

The model produces a higher number of false positives, which contributes to low precision.

This behavior is expected in imbalanced datasets where the model prioritizes detecting positive cases.

## Key Insight

Segmentation-based ROI extraction improves classification performance by focusing the classifier on lesion-relevant regions and reducing background noise.

## Sample Outputs

Generated visual outputs are saved under `outputs/figures/`, including:
## 📊 Visual Results

### ROC Curve
| ROI Model | Full Image Model |
|----------|----------------|
| ![](./outputs/figures/roc_curve_roi.png) | ![](./outputs/figures/roc_curve_full.png) |

---

### Precision-Recall Curve
| ROI Model | Full Image Model |
|----------|----------------|
| ![](./outputs/figures/pr_curve_roi.png) | ![](./outputs/figures/pr_curve_full.png) |

---

### Confusion Matrix
| ROI Model | Full Image Model |
|----------|----------------|
| ![](./outputs/figures/confusion_matrix_roi.png) | ![](./outputs/figures/confusion_matrix_full.png) |

---

### Grad-CAM
| ROI Model | Full Image Model |
|----------|----------------|
| ![](./outputs/figures/gradcam_roi.png) | ![](./outputs/figures/gradcam_full.png) |
- `roc_curve_roi.png`, `roc_curve_full.png`
- `pr_curve_roi.png`, `pr_curve_full.png`
- `confusion_matrix_roi.png`, `confusion_matrix_full.png`
- `gradcam_roi.png`, `gradcam_full.png`
- Training history curves (loss and accuracy/F1) from classification runs

## How to Run (Step-by-Step)

1. Clone repository and enter project folder.
2. Create and activate a Python environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set your dataset and output paths in `config.py`.
5. Run segmentation training + ROI generation:
   ```bash
   python train_segmentation.py
   ```
6. Run classification experiments (ROI + full-image baseline):
   ```bash
   python train_classification.py
   ```
7. (Optional) Generate evaluation plots from saved `.pth` checkpoints without retraining:
   ```bash
   python generate_evaluation_plots.py
   ```

## Where Files Are Stored

- **Checkpoints** are stored in: `checkpoints/`
  - Example: `unet_best.pth`, `efficientnet_b0_roi_best.pth`
- **Outputs** are stored in: `outputs/`
  - Figures: `outputs/figures/`
  - Reports/CSVs: `outputs/reports/`
  - ROI crops: `outputs/roi_crops/`

## Project Structure

```text
PE2_Project/
├── config.py
├── dataset.py
├── model_unet.py
├── utils.py
├── train_segmentation.py
├── train_classification.py
├── generate_evaluation_plots.py
├── PROJECT_EXPLANATION_VIVA.md
├── requirements.txt
├── .gitignore
├── checkpoints/              # saved model weights (.pth)
└── outputs/
    ├── figures/              # ROC, PR, CM, Grad-CAM, curves
    ├── reports/              # manifests and metrics CSVs
    └── roi_crops/            # segmentation-guided crops
```

## Technologies Used

- Python
- PyTorch, Torchvision
- Albumentations
- OpenCV
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- TQDM

## Future Improvements

- K-fold cross-validation for stronger statistical confidence.
- Advanced segmentation backbones (Attention U-Net / DeepLabV3+).
- Test-time augmentation and model ensembling.
- Threshold calibration for clinical sensitivity targets.
- External validation on additional dermoscopy datasets.

