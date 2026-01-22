import torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

MODEL_PATH = "./models/medgemma-1.5-4b-it"
TRAIN_JSON = "./dataset/train.json"
VAL_JSON = "./dataset/val.json"
OUTPUT_DIR = "./outputs/medgemma-lora"

print("Loading base model and processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    use_fast=False
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

# --------------------
# Apply LoRA
# --------------------
print("Applying LoRA...")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --------------------
# Load Dataset
# --------------------
print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_JSON,
        "validation": VAL_JSON,
    },
)

# --------------------
# Preprocess (Gemma3-compatible)
# --------------------
def preprocess(example):
    image = Image.open(example["image"]).convert("RGB")
    answer = example["caption"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "You are a medical imaging expert. "
                        "Describe the findings in this medical image."
                    ),
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Encode prompt + image
    prompt_inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    # Encode answer only (text-only)
    answer_inputs = processor.tokenizer(
        answer,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        add_special_tokens=False,
    )

    input_ids = torch.cat(
        [prompt_inputs["input_ids"], answer_inputs["input_ids"]],
        dim=1,
    )

    attention_mask = torch.cat(
        [
            prompt_inputs["attention_mask"],
            torch.ones_like(answer_inputs["input_ids"]),
        ],
        dim=1,
    )

    labels = torch.cat(
        [
            torch.full_like(prompt_inputs["input_ids"], -100),
            answer_inputs["input_ids"],
        ],
        dim=1,
    )

    return {
        "input_ids": input_ids.squeeze(0),
        "attention_mask": attention_mask.squeeze(0),
        "labels": labels.squeeze(0),
        "pixel_values": prompt_inputs["pixel_values"].squeeze(0),
    }

dataset = dataset.map(
    preprocess,
    remove_columns=dataset["train"].column_names,
)

# --------------------
# Data collator (Gemma3 / LoRA) – FIXED
# --------------------
def data_collator(features):
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f["input_ids"]) for f in features],
            batch_first=True,
            padding_value=processor.tokenizer.pad_token_id,
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f["attention_mask"]) for f in features],
            batch_first=True,
            padding_value=0,
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f["labels"]) for f in features],
            batch_first=True,
            padding_value=-100,
        ),
        "pixel_values": torch.stack([f["pixel_values"] if isinstance(f["pixel_values"], torch.Tensor) else torch.tensor(f["pixel_values"]) for f in features]),
    }


# --------------------
# Training
# --------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=1,
    save_steps=50,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,   # important for custom forward
    label_names=["labels"],        # important for Trainer
    eval_strategy="steps",
    eval_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
)

print("🚀 Starting training...")
trainer.train()

print("✅ Training complete!")
model.save_pretrained("./medgemma-lora-output")
