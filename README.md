# 🩺 DeepVesselNet for Retina Blood Vessel Segmentation

This repository contains a **DeepVesselNet** implementation for **retinal blood vessel segmentation** using fundus images.  
It was developed for **medical image analysis research** and optimized to detect both large and fine retinal vessels.

---

## 📌 Features

- **DeepVesselNet architecture** for vessel segmentation
- **Thin-vessel enhancement** using:
  - Patch-based training
  - Focal Tversky loss with recall weighting
  - Contrast Limited Adaptive Histogram Equalization (CLAHE)
- **Post-processing** to clean segmentation maps
- **High-resolution training** (512×512 with patch size 256)
- **Evaluation metrics**: Dice, IoU, Sensitivity, Specificity
- **Visualization utilities** for paper-ready figures

---

## 📂 Project Structure

DeepVesselNet/
│
├── Data/ # Dataset folder (not included in repo)
│ ├── train/image/ # Training images
│ ├── train/mask/ # Training masks
│ ├── test/image/ # Test images
│ ├── test/mask/ # Test masks
│
├── outputs/ # Model outputs
│ ├── checkpoints/ # Saved models (.pth)
│ ├── predictions/ # Predicted segmentation masks
│ ├── loss_curve.png # Training/validation loss curves
│
├── src/ # Source code
│ ├── model.py # DeepVesselNet architecture
│ ├── data_loader.py # Data pipeline with augmentation
│ ├── loss.py # Loss functions (Tversky, Focal Tversky)
│ ├── train.py # Training script
│ ├── evaluate.py # Evaluation script
│
├── notebooks/ # Jupyter notebooks for EDA/experiments
│
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── LICENSE

yaml
Copy
Edit

---

## 📊 Example Results

### Original Image → Ground Truth → Prediction

![Example](outputs/example_result.png)

**Evaluation (GPU trained, thin-vessel optimized):**
| Metric | Score |
|--------------|---------|
| Dice | 0.8087 |
| IoU | 0.6790 |
| Sensitivity | 0.8357 |
| Specificity | 0.9783 |

---

## ⚙️ Installation

```bash
# Clone repo
git clone https://github.com/Adith-Sree-Jey/DeepVesselNet-Retina-Segmentation.git
cd DeepVesselNet-Retina-Segmentation

# Create and activate virtual environment
python -m venv deepvessel_env
.\deepvessel_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```
