import torch
from PIL import Image
import gradio as gr
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# --------------------
# Paths
# --------------------
BASE_MODEL_PATH = "./models/medgemma-1.5-4b-it"
LORA_MODEL_PATH = "./medgemma-lora-output"

# --------------------
# Load model and processor
# --------------------
processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, use_fast=False)
base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
model = model.to(torch.float32)
model.eval()

# --------------------
# Prediction function
# --------------------
def predict(image: Image.Image):
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
        images=image.convert("RGB"),
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model.generate(
            **{k: v.to(model.device) for k, v in inputs.items()},
            max_new_tokens=128,
            do_sample=False
        )

    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# --------------------
# Gradio interface
# --------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(
        label="Predicted Caption",
        lines=15,        # initial visible height
        max_lines=1000,  # allows scroll for long text
        interactive=False,
        placeholder="The caption will appear here..."
    ),
    title="MedGemma LoRA - Medical Image Captioning",
    description="Upload a chest X-ray, MRI, or CT scan and get a medical description."
)



demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
