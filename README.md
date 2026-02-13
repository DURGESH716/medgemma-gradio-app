# 🏥 MedGemma LoRA: Clinical Medical Image Captioning

### **Architecture:** Fine-tuned Google MedGemma 1.5–4B (Vision-Language Model)

### **Methodology:** Parameter-Efficient Fine-Tuning (PEFT) via LoRA

## 🎯 Problem Statement

Interpretive medical imaging (X-ray, MRI, CT) often suffers from a clinical "reporting bottleneck." While deep learning models can classify diseases, they rarely provide the descriptive narrative required for standard medical records.

**Our Solution:** This project fine-tunes **MedGemma**—a specialized medical VLM—to automate the generation of clinically styled captions. By leveraging **LoRA**, the system learns to analyze raw visual tokens and synthesize a diagnostic description, assisting in rapid clinical documentation and preliminary screening.


## 📊 Data Information & Preparation

The system utilizes a structured dataset where medical images are paired with ground-truth expert captions.

### **Dataset Structure**

* **Manifests:** Data is managed via `dataset/train.json` and `dataset/val.json`.
* **Format:** Each entry maps a local image path to its corresponding clinical description.
* **JSON Entry Example:**

```json
{
  "image": "data/images/xray1.jpeg",
  "caption": "The chest X-ray shows no focal consolidation, pleural effusion, or pneumothorax."
}

```

## Model Architecture

1. **Vision Encoder:** Utilizes the SigLIP-based vision tower to extract high-resolution spatial features from medical scans.
2. **Multimodal Adapter:** A learned projection layer that maps visual features into the language model's embedding space.
3. **Language Backbone:** The MedGemma 4B transformer, pre-aligned by Google for medical terminology and clinical reasoning.

<p align="center">
<img width="916" height="600" alt="image" src="https://github.com/user-attachments/assets/8c2e32fa-2e3b-4013-8750-cb3ff06c64bf" />
<em> Figure-1: Unified Architecture featuring Vision Encoder, Adapter, and Use-case </em>
</p>

## ⚙️ Training Parameters (LoRA)

We implement **LoRA (Low-Rank Adaptation)** to enable high-fidelity fine-tuning on consumer-grade hardware, reducing VRAM consumption while maintaining medical accuracy.

| Parameter | Value | Description |
| --- | --- | --- |
| **Base Model** | `medgemma-1.5-4b-it` | Google's medical-specific instruction model. |
| **Precision** | `bfloat16` | Optimized for NVIDIA Ampere/Blackwell Tensor Cores. |
| **Rank ()** | 16 | Dimension of the low-rank update matrices. |
| **Target Modules** | `q_proj, v_proj` | Targeted attention projections for adaptation. |


## 📂 Repository Structure

```text
.
├── train_.py                   # LoRA fine-tuning script
├── create_real_dataset.py      # JSON text format script
├── test_lora.py                # Inference test script
├── gradio_app.py               # Gradio deployment app
├── dataset/
│   ├── train.json
│   └── val.json
├── data/
│   └── images/                 # Consists sample medical images
├── outputs/
│   └── medgemma-lora/          # LoRA adapter output (created after training)
├── requirements.txt
└── README.md

```

## 🚀 Getting Started (Replication)

### 1. Setup Environment

Ensure you are using Python 3.10+ on a Linux environment with CUDA 12.8.

```bash
python3 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt

```

### 2. Train LoRA Adapter

Run the fine-tuning script to adapt the base MedGemma model to your dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python train_.py

```

### 3. Launch Gradio App

Deploy the interactive diagnostic interface:

```bash
CUDA_VISIBLE_DEVICES=0 python gradio_app.py

```

## Challenges & Solutions

* **VRAM Constraints:** Fine-tuning a 4B model usually requires significant memory; we utilized **LoRA** and **BFloat16** to fit the training within a single 16GB+ GPU.
* **Structural Consistency:** Managed file pathing issues across Linux environments by implementing a robust `create_real_dataset.py` utility.
* **Clinical Accuracy:** Vanilla models often use colloquial terms; MedGemma's medical-specific pre-training ensures "radiology-ready" terminology.

## 🔮 Future Scope

* **DICOM Compatibility:** Enabling direct ingestion of `.dcm` medical imaging files.
* **Bilingual Reporting:** Supporting multi-language clinical caption generation for global deployment.
* **Model Quantization:** Using 4-bit NF4 quantization to deploy the 4B model on edge-computing screening devices.
