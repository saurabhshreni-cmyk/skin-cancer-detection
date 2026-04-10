# Skin Cancer Detection using Segmentation + Classification

## 🌐 Live Demo
👉 https://skin-cancer-detection-saurabh.streamlit.app

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red)
![Dataset](https://img.shields.io/badge/Dataset-ISIC-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🧠 Overview

End-to-end deep learning pipeline for skin lesion analysis:

`Image → Segmentation → ROI → Classification → Prediction + Grad-CAM`

---

## 🎯 Demo Features

- Upload dermoscopic image
- ROI vs Full-image prediction
- Grad-CAM visualization
- Real-time inference

---

## 📊 Results

| Model | Dice Score | ROC-AUC |
|------|-----------|--------|
| U-Net (Segmentation) | ~0.85 | — |
| ROI Classifier | — | 0.8746 |
| Full Image Classifier | — | 0.8793 |

---

## ⚠️ Key Insights

- Full-image model slightly outperforms ROI model
- ROI improves interpretability (better Grad-CAM focus)
- Class imbalance affects precision

---

## 🔍 Visual Results

### ROC Curve
| ROI Model | Full Model |
|----------|------------|
| ![](assets/roc_curve_roi.png) | ![](assets/roc_curve_full.png) |

### Precision-Recall Curve
| ROI Model | Full Model |
|----------|------------|
| ![](assets/pr_curve_roi.png) | ![](assets/pr_curve_full.png) |

### Confusion Matrix
| ROI Model | Full Model |
|----------|------------|
| ![](assets/confusion_matrix_roi.png) | ![](assets/confusion_matrix_full.png) |

### Grad-CAM
| ROI Model | Full Model |
|----------|------------|
| ![](assets/gradcam_roi.png) | ![](assets/gradcam_full.png) |

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
