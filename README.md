
# MedGemma LoRA – Medical Image Captioning (Gradio App)

This project fine-tunes **Google MedGemma 1.5–4B (Image + Text)** using **LoRA**
for medical image captioning and deploys it via **Gradio**.

The application accepts medical images (X-ray, MRI, CT) and generates
clinically styled descriptive captions.


---

## System Requirements

- Linux (Ubuntu recommended)
- Python ≥ 3.10
- NVIDIA GPU (recommended, 16GB+ VRAM)
- CUDA 12.8
- Internet access (for first-time model download)

---

## Repository Structure

```

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

````

---

## Step 1: Create Virtual Environment

```bash
python3 -m venv .venv311
source .venv311/bin/activate
````

---

## Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 3: Dataset Preparation

Your dataset files must already exist:

* `dataset/train.json`
* `dataset/val.json`

Each JSON entry should follow this format:

```json
{
  "image": "data/images/xray1.jpeg",
  "caption": "The chest X-ray shows no focal consolidation..."
}
```

Image paths must be relative paths and point to real image files.

---

## Step 4: Train LoRA Adapter

Run this **only if you want to fine-tune** MedGemma on your dataset.

```bash
CUDA_VISIBLE_DEVICES=0 python train_.py
```

What happens:

* Base model `google/medgemma-1.5-4b-it` is downloaded automatically
* LoRA adapters are trained
* Output is saved to:

```
outputs/medgemma-lora/
├── adapter_config.json
└── adapter_model.safetensors
```

---

## Step 5: Test LoRA Inference

Verify the trained adapter using a single image:

```bash
CUDA_VISIBLE_DEVICES=0 python test_lora.py
```

Expected output:

```
Predicted Caption:
Based on the medical image, the lungs appear...
```

This confirms:

* Base model loading
* LoRA adapter loading
* Image + text generation works correctly

---

## Step 6: Launch Gradio App

Start the web interface:

```bash
CUDA_VISIBLE_DEVICES=0 python gradio_app.py
```

You will see:

```
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://xxxx.gradio.live
```

Open the link, upload a medical image, and receive a caption.
