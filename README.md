# 🏗️ Critical Infrastructure Detection using RF-DETR

> Detect and localize 19 types of critical infrastructure in aerial imagery using state-of-the-art Transformer-based object detection.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 About The Project

This project implements **Critical Infrastructure Detection (CID)** using the state-of-the-art **RF-DETR** (Receptive Field DETR) model. The system identifies and localizes **19 types** of critical infrastructure assets in aerial and satellite imagery, including:

- 🛫 Airport Runways
- 🌉 Bridges  
- ⚡ Power Plants (Nuclear, Thermal, Solar)
- 🌊 Dams
- 🏭 Oil Refineries
- And 14 more infrastructure types

**Why RF-DETR?** Unlike traditional CNN-based detectors (like YOLO), RF-DETR uses **Transformer attention mechanisms** to capture global context—essential for detecting large-scale infrastructure in complex terrain.

### ✨ Key Features

- ✅ **End-to-End Pipeline**: From data collection to deployment
- ✅ **Transformer Architecture**: Superior accuracy for large objects
- ✅ **19 Infrastructure Classes**: Comprehensive coverage
- ✅ **Batch Processing**: Efficient inference on multiple images
- ✅ **Visual Results**: Automatic bounding box visualization with confidence scores

---

## 📂 Dataset & Data Generation

The model is trained on a **custom-curated dataset** built from scratch using a rigorous 4-step process:

### 1️⃣ Manual Data Collection
Diverse aerial and satellite imagery gathered from multiple sources and geographical regions.

### 2️⃣ Synthetic Image Generation
Enhanced training data using **Nano Banana Pro** and **Seedream 4** to handle edge cases and improve model robustness.

### 3️⃣ Precise Annotation
Manual annotation using **Label Studio** with pixel-perfect bounding boxes for high-quality ground truth.

### 4️⃣ Data Cleaning & Preprocessing
**ChatGPT-assisted** dataset cleaning to standardize annotations and filter low-quality samples.

### 🏷️ Supported Infrastructure Classes (19 Total)

```
Airport_Runway               Bridge                    Cargo_Ship
Cooling_Tower                Dam                       Electrical_Substation
Energy_Storage_Infrastructure Mobile_Tower              Nuclear_Reactor
Oil_Refinery                 Satellite_Dish            Seaport
Shipping_Containers          Solar_Power_Plant         Thermal_Power_Plant
Transmission_Tower           Water_Tower               Wind_Turbine
Mobile_Harbour_Cranes
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/RF-DETR-Infrastructure-Detection.git
cd RF-DETR-Infrastructure-Detection

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

**Train the model:**
```bash
python src/train.py \
    --dataset_dir ./data/cid_dataset \
    --output_dir ./output \
    --epochs 50 \
    --batch_size 4
```

**Run inference on images:**
```bash
python src/predict.py \
    --checkpoint ./output/checkpoint_best_ema.pth \
    --source ./test_images \
    --output ./results \
    --conf 0.35
```

**Process a single image:**
```bash
python src/predict.py \
    --checkpoint ./output/checkpoint_best_ema.pth \
    --image ./my_image.jpg \
    --output ./results \
    --conf 0.35 \
    --visualize
```

---

## 📊 Results

Performance metrics on validation dataset:

| Class | Precision | Recall | mAP@50 |
|-------|-----------|--------|--------|
| Dam | 0.85 | 0.82 | 0.84 |
| Bridge | 0.79 | 0.76 | 0.78 |
| Airport Runway | 0.88 | 0.85 | 0.87 |
| Nuclear Reactor | 0.91 | 0.89 | 0.90 |
| Solar Power Plant | 0.83 | 0.80 | 0.82 |
| **Overall (19 classes)** | **0.81** | **0.78** | **0.80** |

*Note: Replace with your actual evaluation results*

---

## 📁 Project Structure

```
RF-DETR-Infrastructure-Detection/
├── src/
│   ├── train.py              # Training script
│   ├── predict.py            # Inference script
│   ├── evaluate.py           # Evaluation metrics
│   └── models/               # Model architectures
├── data/                     # Dataset directory (COCO format)
├── configs/                  # Configuration files
├── notebooks/                # Jupyter notebooks
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU training)
- 8GB+ RAM (16GB recommended)

See `requirements.txt` for full dependency list.

---

## 🎯 Use Cases

- 🏛️ **Defense & Security**: Critical infrastructure monitoring
- 🗺️ **Urban Planning**: Infrastructure mapping and assessment
- 🚨 **Disaster Response**: Rapid damage assessment after natural disasters
- 📊 **Research**: Geospatial analysis and infrastructure studies

---

## 🤝 Acknowledgements

This project builds upon:

- [Original RF-DETR Paper](https://arxiv.org/abs/2303.10845) & [GitHub Repository](https://github.com/liming-ai/RF-DETR)
- [Roboflow](https://roboflow.com/) for dataset management tools
- [Supervision](https://github.com/roboflow/supervision) library for visualization
- [Label Studio](https://labelstud.io/) for annotation platform
- [PyTorch](https://pytorch.org/) deep learning framework

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

**Your Name** - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/YOUR_USERNAME/RF-DETR-Infrastructure-Detection](https://github.com/YOUR_USERNAME/RF-DETR-Infrastructure-Detection)

---

<div align="center">

⭐ **Star this repo if you find it useful!** ⭐

*Built with ❤️ for critical infrastructure monitoring*

</div>
