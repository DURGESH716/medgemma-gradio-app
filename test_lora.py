import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# --------------------
# Paths
# --------------------
BASE_MODEL_PATH = "./models/medgemma-1.5-4b-it"
LORA_MODEL_PATH = "./medgemma-lora-output"
IMAGE_PATH = "./data/images/xray1.jpeg"  # Change to your test image

# --------------------
# Load base model and processor
# --------------------
print("Loading base model and processor...")

processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, use_fast=False)

base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

# --------------------
# Apply LoRA adapter
# --------------------
print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
model = model.to(torch.float32)  # ✅ Convert to FP32 for stable generation
model.eval()

# --------------------
# Prepare input
# --------------------
image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": "You are a medical imaging expert. Describe the findings in this medical image."
            },
        ],
    }
]

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)

# --------------------
# Generate caption
# --------------------
with torch.no_grad():
    outputs = model.generate(
        **{k: v.to(model.device) for k, v in inputs.items()},
        max_new_tokens=128,  # prevent extremely long sequences
        do_sample=False       # deterministic output
    )

# --------------------
# Decode output
# --------------------
answer = processor.decode(outputs[0], skip_special_tokens=True)
print("Predicted Caption:")
print(answer)
