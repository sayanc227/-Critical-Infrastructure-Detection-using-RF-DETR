# 🏗️ Critical Infrastructure Detection using RF-DETR

Transformer-based RF-DETR model for drone & aerial surveillance detecting critical infrastructure (bridges, power plants, substations, towers).

<div align="center">

<table>
  <tr>
    <td width="50%">
      <img src="https://github.com/user-attachments/assets/f2774356-3e9f-4079-af99-945f2cdccb0a" alt="Detection Example 1"/>
    </td>
    <td width="50%">
      <img src="https://github.com/user-attachments/assets/b080e1eb-a57c-46cc-bc8b-00a93e4fa179" alt="Detection Example 2"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Large-scale infrastructure detection from drone footage</em>
    </td>
    <td align="center">
      <em>Simultaneous multi-class detection in dense environments</em>
    </td>
  </tr>
</table>

</div>


### 🎯 Detect and localize 19 types of critical infrastructure in aerial imagery
*Using state-of-the-art RF-DETR (Transformer-based object detection)*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![RF-DETR](https://img.shields.io/badge/Model-RF--DETR-orange.svg)]()
[![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-purple.svg)](https://roboflow.com/)
[![Label Studio](https://img.shields.io/badge/Label%20Studio-Annotation-ff6b6b.svg)](https://labelstud.io/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/167xgsFcpFqzVfAbT88IFdUYeM_X0h1um?usp=sharing)
[![Batch Processing](https://img.shields.io/badge/Batch-Inference-blue.svg)]()
[![mAP@50](https://img.shields.io/badge/mAP@50-76%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[🚀 Quick Start](#quick-start) • [📊 Results](#results)

</div>

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

### 1. Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/sayanc227/Critical-Infrastructure-Detection-using-RF-DETR.git
cd Critical-Infrastructure-Detection-using-RF-DETR
pip install -r requirements.txt
```

> **Note:** A CUDA-enabled GPU is strongly recommended. These scripts are compatible with Google Colab and local GPU machines.

---

### 2. Dataset Structure

The dataset must follow COCO format and be organized as:

```
dataset/
├── train/
│   ├── images...
│   └── _annotations.coco.json
├── valid/
│   ├── images...
│   └── _annotations.coco.json
└── test/
    ├── images...
    └── _annotations.coco.json
```

---

### 3. Training the Model

To start training from scratch:

```bash
python train.py \
  --dataset_dir path/to/dataset \
  --output_dir output/ \
  --epochs 50 \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --lr 1e-4 \
  --early_stopping
```

To resume training from a checkpoint:

```bash
python train.py \
  --dataset_dir path/to/dataset \
  --output_dir output/ \
  --resume output/checkpoint0049.pth \
  --epochs 65
```

---

### 4. Evaluating a Trained Model

Evaluate a trained model using a saved checkpoint:

```bash
python evaluate.py \
  --dataset_dir path/to/dataset \
  --checkpoint output/checkpoint_best_total.pth
```

This will compute COCO-style metrics (mAP, AP@0.5, AP@0.75, recall) and optionally save predictions for further analysis.

---

### 5. Running Inference (Prediction)

Run inference on a single image:

```bash
python predict.py \
  --dataset_dir path/to/dataset \
  --checkpoint output/checkpoint_best_total.pth \
  --image_path path/to/image.jpg \
  --threshold 0.3
```

Output includes detected class IDs, confidence scores, and bounding box coordinates. The confidence threshold can be adjusted based on operational requirements.


---

For any deployment, customization, or integration needs, please contact the project author.

## 📊 Results
## RF-DETR Validation Performance Metrics

### 📊 Metrics Overview
- **Precision**: The accuracy of positive predictions.  
- **Recall**: The ability to find all positive instances.  
- **$mAP_{50}$**: Mean Average Precision at an Intersection over Union (IoU) threshold of 0.50.  
- **$mAP_{50:95}$**: Average mAP across IoU thresholds from 0.50 to 0.95 (COCO standard).  

This table summarizes the performance of your model across all 19 infrastructure classes on the validation set.

| Class                           | Precision | Recall | mAP@50 | mAP@50:95 |
|---------------------------------|-----------|--------|--------|-----------|
| **All Classes (Avg)**           | **0.747** | **0.630** | **0.761** | **0.506** |
| Water Tower                     | 1.000     | 0.630  | 1.000  | 0.897     |
| Electrical Substation           | 1.000     | 0.630  | 1.000  | 0.545     |
| Oil Refinery                    | 1.000     | 0.630  | 0.996  | 0.869     |
| Airport Runway                  | 1.000     | 0.630  | 0.847  | 0.387     |
| Wind Turbine                    | 1.000     | 0.630  | 0.904  | 0.635     |
| Transmission Tower              | 0.977     | 0.630  | 0.895  | 0.696     |
| Satellite Dish / Ground Station | 0.966     | 0.630  | 0.945  | 0.802     |
| Bridge                          | 0.944     | 0.630  | 0.909  | 0.616     |
| Mobile Tower                    | 0.938     | 0.630  | 0.828  | 0.563     |
| Energy Storage Infrastructure   | 0.887     | 0.630  | 0.854  | 0.633     |
| Nuclear Reactor                 | 0.854     | 0.630  | 0.753  | 0.508     |
| Seaport                         | 0.842     | 0.630  | 0.825  | 0.537     |
| Thermal Power Plant             | 0.818     | 0.630  | 0.767  | 0.432     |
| Dam                             | 0.621     | 0.630  | 0.647  | 0.251     |
| Solar Power Plant               | 0.379     | 0.630  | 0.578  | 0.401     |
| Cooling Tower                   | 0.333     | 0.630  | 0.333  | 0.250     |
| Cargo Ship                      | 0.318     | 0.630  | 0.512  | 0.152     |
| Mobile Harbour Cranes            | 0.283     | 0.630  | 0.509  | 0.320     |
| Shipping Containers             | 0.035     | 0.630  | 0.354  | 0.114     |
            

---

## 📁 Project Structure

```
Critical-Infrastructure-Detection-using-RF-DETR/
├── assets/              # Visual results & UI assets
├── src/                 # Training, inference, evaluation code
├── notebooks/           # Colab / Jupyter notebooks
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── LICENSE              # License
└── .gitignore           # Git ignore rules


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

**Sayan C** - [13sayanc@gmail.com](mailto:your.email@example.com)

Project Link: [https://github.com/sayanc227/RF-DETR-Infrastructure-Detection](https://github.com/sayanc227/Critical-Infrastructure-Detection-using-RF-DETR)

---

<div align="center">

⭐ **Star this repo if you find it useful!** ⭐

</div>
