# ♻️ Edge AI Model for Real-Time Waste Classification

## 🎯 Project Summary

This document serves as the master report for the Edge AI Prototype (Part 2, Task 1). The project successfully implemented a lightweight image classification pipeline using the **MobileNetV2** architecture ($\alpha=0.35$), optimized with **Full Integer Quantization (Int8)** for low-latency deployment on edge devices like the Raspberry Pi.

***

## 📊 Practical Implementation: Metrics & Architecture

This table summarizes the core architectural choices and the final performance metrics achieved after training and optimization, fulfilling Part 2, Task 1.

| Metric | Keras Model (Float32 Baseline) | TFLite Model (Int8 Final) | Edge Benefit |
| :--- | :--- | :--- | :--- |
| **Model** | MobileNetV2 ($\alpha=0.35$) | MobileNetV2 ($\alpha=0.35$) | Low parameter count for fast execution. |
| **Optimization**| Standard Keras H5 | **Full Integer Quantization (Int8)**  | **$4x$ speed boost** and **$75\%+$ size reduction** on Raspberry Pi CPU. |
| **Accuracy** | **88.0%** | **87.5%** | Minimal accuracy drop after compression. |
| **Model Size** | 6.5 MB | **710 KB** | Maximizes available RAM on the edge device. |
| **Inference Latency** | N/A (Host-Only Test) | **20 ms** | Guarantees real-time decision-making speed. |

---

## ⚡ Edge AI Value Proposition (Theoretical Benefits)

Edge AI, the core of this project, delivers critical benefits over traditional cloud-based systems for real-time applications, fulfilling the "Explain how Edge AI benefits" requirement:

1.  **Minimal Latency (Near-Instant Decisions):** Processing occurs locally, eliminating network latency. This guarantees **sub-100ms inference times**, enabling immediate actions such as activating a sorting mechanism.
2.  **Operational Reliability (Internet Independence):** The core decision-making loop is **immune to network outages** or slow connections, ensuring continuous operation and high system uptime.
3.  **Data Privacy and Security:** Raw sensor data never leaves the local environment. Only non-sensitive results are transmitted, enhancing **security and compliance**.
4.  **Reduced Bandwidth and Cost:** Only small classification results are sent to the central management system, drastically **reducing data transmission costs**.

---

## 📦 Deployment Instructions (Using `app.py`)

The final inference script, `app.py`, runs on the Raspberry Pi using the optimized **`tflite-runtime`**.

### Setup and Installation
1.  Clone this repository to your Raspberry Pi.
2.  Install the lightweight dependencies (essential for low memory use):
    ```bash
    pip install tflite-runtime numpy pillow
    ```
3.  Ensure a sample image named `test_image.jpg` is available in the root directory.

### Running Inference
Execute the main application file:
python3 app.py

**Files Contained in Repository**
File/Folder	Purpose
TFLite.py	Training, Conversion, and Quantization script
app.py -	Deployment and Real-time Inference on RPi
README.md	-This Master Report document
deployment_package/-	Final models and configuration files
deployment_package/*.tflite	-Optimized TensorFlow Lite model
deployment_package/*.h5-	Backup Keras model
deployment_package/class_indices.json	-Label mappings

**📄Full Assignment Deliverables Reference**
The remaining theoretical and conceptual parts of the assignment are completed and stored in Microsoft Word (.docx) format.

📍 Google Drive Link: ['https://drive.google.com/drive/folders/1Aw6d7MTgAVTRi3FVAcKDpAeUJtaiev4O?usp=drive_link']

📋 **Document Overview**
Assignment Part	Required Deliverable	Document Title
Part 1	Theoretical Analysis (Q1, Q2, Q3) & Case Study	Part_1_Theoretical_Analysis.docx
Part 2, Task 2	AI-Driven IoT Concept (Proposal & Diagram)	Part_2_Task_2_AgriConcept.docx
Part 2, Task 3	Ethics in Personalized Medicine (300-word analysis)	Part_2_Task_3_Ethics_Analysis.docx
Part 3	Futuristic Proposal (Concept Paper)	Part_3_Futuristic_Proposal.docx
🔗 Quick Links
[📁 Google Drive Folder](['https://drive.google.com/drive/folders/1Aw6d7MTgAVTRi3FVAcKDpAeUJtaiev4O?usp=drive_link']) - All theoretical deliverables

🐍 Python Scripts - Implementation code

📦 Deployment Package - Optimized models & assets

<div align="center">
🚀 Edge AI Waste Classification System
Real-time processing for sustainable waste management

</div> ```