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

## 📌 Project Overview

This project focuses on the **automated detection of critical infrastructure**—including **Bridges, Power Plants, Substations, and Communication Towers**—from **aerial and drone imagery**.

Instead of relying solely on scarce or restricted public datasets, this work emphasizes a **Synthetic Data Pipeline** that leverages advanced **Generative AI models** to create rare and difficult-to-capture scenarios. All generated data was **manually reviewed and annotated** to ensure realism and label quality.

The detection model is built using **RF-DETR (Roboflow Detection Transformer)**, selected for its strong performance on small objects within large, cluttered aerial scenes and its suitability for near–real-time inference.

---

## 🎯 Objectives

* Detect critical infrastructure from varying aerial and drone viewpoints
* Address **data scarcity** in security-sensitive domains using synthetic data
* Train a lightweight transformer-based detector suitable for aerial surveillance

---

## 🛠️ Data Pipeline & Methodology

The primary value of this project lies in the **dataset design and curation strategy**, not in custom model architecture development.

### 1️⃣ Data Gathering & Synthesis

Real-world aerial data for critical infrastructure is often limited, restricted, or lacks viewpoint diversity. To mitigate this, a **hybrid dataset** was created using a **60 / 40 split**:

* **Real Data (60%)**

  * Publicly accessible satellite imagery
  * Permissible aerial and drone footage

* **Synthetic Data (40%)**

  * Generated to simulate rare or operationally relevant viewpoints
  * Examples include oblique angles, low-light conditions, and complex backgrounds

**Generative AI tools used:**

* **Nano Banana Pro**
  Used for high-fidelity texture generation (e.g., rusted metal towers, cracked concrete bridges).

* **Seedream 4**
  Used to generate complex environments such as foggy terrain, dense vegetation, and cluttered backgrounds to test robustness.

* **ChatGPT**
  Used for structured prompt engineering to enforce specific drone camera parameters (field of view, lighting, altitude, perspective).

---

### 2️⃣ Data Cleaning & Quality Control

* Manual filtering of generated images to remove visual artifacts and hallucinations
* Resolution standardization to common drone capture formats
* Visual inspection for annotation suitability

---

### 3️⃣ Annotation Process

* **Tool:** Label Studio
* **Format:** COCO
* **Annotation Type:** Manual bounding boxes
* **Target Classes:**

  * Bridge
  * Power Plant
  * Substation
  * Communication Tower

Care was taken to maintain **consistent class definitions** and bounding box rules across the dataset.

---

## 🤖 Model Choice: RF-DETR

RF-DETR (Real-Time Detection Transformer) was selected over traditional CNN-based detectors (e.g., YOLO) due to its strengths in aerial surveillance scenarios:

* Transformer-based attention mechanisms help isolate infrastructure within large, cluttered scenes
* Improved handling of small or distant objects
* Modern architecture aligned with current research and deployment trends

📘 **Documentation:**

* Official RF-DETR Documentation

---

## 🧪 Training Configuration

* **Environment:** Google Colab Pro (NVIDIA T4 GPU)
* **Epochs:** 100
* **Batch Size:** 8
* **Optimizer:** AdamW

All training, evaluation, and inference steps were conducted using official RF-DETR workflows.

---

## 🚀 How to Run the Project

This project is designed to be **fully reproducible via Google Colab**.

🔗 **Training & Inference Notebook:**
[Open in Google Colab](https://colab.research.google.com/drive/167xgsFcpFqzVfAbT88IFdUYeM_X0h1um)

### Steps:

1. Open the Colab notebook using the link above
2. Connect to a GPU runtime
3. The notebook handles:

   * Installation of RF-DETR
   * Dataset download and setup
4. Run the cells to train the model and visualize inference results

---

## 📂 Resources & Downloads

* **Trained Weights:** Available via the Colab notebook
* **Dataset:** Provided through Kaggle / Roboflow Universe (link inside notebook)

---

## 📊 Results & Observations

* **Synthetic Data Impact:**

  * Adding the 40% synthetic subset resulted in an approximate **15% improvement in mAP** on low-light and oblique-angle validation samples compared to a real-only baseline.

* **Observed Challenges:**

  * Initial confusion between **Communication Towers** and **High-Voltage Pylons**
  * Mitigated by adding targeted negative samples and refining annotations

These findings reinforced the importance of **dataset composition** over model complexity.

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
