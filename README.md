# Skin Cancer Detection using Segmentation + Classification

## Live Demo
👉 https://skin-cancer-detection-saurabh.streamlit.app

---

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red)
![Dataset](https://img.shields.io/badge/Dataset-ISIC-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

This project presents an end-to-end deep learning pipeline for skin lesion analysis, combining:

- **Segmentation (U-Net)** — isolates lesion region
- **ROI Extraction** — focuses on lesion area
- **Classification (EfficientNet-B0)** — predicts malignancy
- **Grad-CAM** — explains model decisions

### Pipeline
Image → Segmentation → ROI Extraction → Classification → Prediction + Explainability
---

## Demo Features

- Upload dermoscopic image (JPG/PNG)
- ROI-based and full-image prediction
- Grad-CAM visualization
- Real-time inference via Streamlit UI
- Deployed and accessible online

---

## Dataset

### ISIC 2018 (Segmentation)
- Input: dermoscopic images
- Target: binary lesion masks (`*_segmentation.png`)
- Used for training U-Net segmentation

### ISIC 2020 (Classification)
- Input: dermoscopic images
- Target: binary label (`0 = benign`, `1 = malignant`)
- Used for classification models

---

## Results

| Model | Dice Score | ROC-AUC | Accuracy | Recall | F1 |
|-------|-----------|---------|----------|--------|----|
| U-Net (Segmentation) | 0.85 | — | — | — | — |
| ROI Classifier | — | 0.8746 | 0.8212 | 0.7521 | 0.1293 |
| Full Image Classifier | — | **0.8793** | 0.8015 | 0.8034 | 0.1251 |

---

## Key Observations

- Full-image model marginally outperforms ROI-based model on AUC (0.8793 vs 0.8746)
- ROI model provides better Grad-CAM focus — activations concentrate on the lesion interior
- Low precision on both models is expected due to severe class imbalance (~98% benign in ISIC 2020)
- A fixed threshold of 0.5 on imbalanced data maximises recall at the cost of precision — this is the intended behaviour for a screening tool

---

## Note on Metrics

- **ROC-AUC (~0.88)** is the primary metric — threshold-independent ranking ability
- **Precision is low (~0.07)** due to the 98:2 class imbalance at threshold 0.5
- The model distinguishes classes well; threshold calibration is needed for deployment

### Future Improvements
- Threshold calibration to clinical sensitivity targets
- K-fold cross-validation
- Attention U-Net / DeepLabV3+ for better segmentation
- Test-time augmentation and model ensembling
- External dataset validation

---

## Visual Results

### ROC Curve

<img src="https://raw.githubusercontent.com/saurabhshreni-cmyk/skin-cancer-detection/main/assets/roc_curve_roi.png" width="400">
<img src="https://raw.githubusercontent.com/saurabhshreni-cmyk/skin-cancer-detection/main/assets/roc_curve_full.png" width="400">

---

### Precision-Recall Curve

<img src="https://raw.githubusercontent.com/saurabhshreni-cmyk/skin-cancer-detection/main/assets/pr_curve_roi.png" width="400">
<img src="https://raw.githubusercontent.com/saurabhshreni-cmyk/skin-cancer-detection/main/assets/pr_curve_full.png" width="400">

---

### Confusion Matrix

<img src="https://raw.githubusercontent.com/saurabhshreni-cmyk/skin-cancer-detection/main/assets/confusion_matrix_roi.png" width="400">
<img src="https://raw.githubusercontent.com/saurabhshreni-cmyk/skin-cancer-detection/main/assets/confusion_matrix_full.png" width="400">

---

### Grad-CAM Visualization

<img src="https://raw.githubusercontent.com/saurabhshreni-cmyk/skin-cancer-detection/main/assets/gradcam_roi.png" width="400">
<img src="https://raw.githubusercontent.com/saurabhshreni-cmyk/skin-cancer-detection/main/assets/gradcam_full.png" width="400">

---

## Run Locally

```bash
git clone https://github.com/saurabhshreni-cmyk/skin-cancer-detection.git
cd skin-cancer-detection
pip install -r requirements.txt
streamlit run app.py
```

---

## Inference (CLI)

```bash
python predict.py --image_path "path/to/image.jpg" --mode both
```

Modes: `roi` | `full` | `both`

---

## Project Structure

skin-cancer-detection/
├── app.py # Streamlit web app
├── predict.py # Inference script
├── config.py # Configuration settings
├── dataset.py # Dataset loader
├── model_unet.py # U-Net model (segmentation)
├── train_segmentation.py # Train segmentation model
├── train_classification.py # Train classification model
├── generate_evaluation_plots.py# Generate ROC, PR, CM plots
├── requirements.txt # Dependencies

├── checkpoints/ # Saved model weights (.pth)
├── assets/ # Images used in README (GitHub-visible)
└── outputs/ # Generated results (ignored in Git)

---

## Technologies Used

- Python 3.10
- PyTorch 2.2, Torchvision
- OpenCV, Albumentations
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- Streamlit

---

## Disclaimer

This project is for educational purposes only and is not intended for medical diagnosis.

---

## Author

**Saurabh Dwadash Shreni**  
B.Tech CSE (AI), Manipal Institute of Technology, Bengaluru  
[GitHub](https://github.com/saurabhshreni-cmyk)
