import os, json
from sklearn.model_selection import train_test_split

DATA_DIR = "./data/images"
ALL_IMAGES = os.listdir(DATA_DIR)

captions = {
    "xray1.jpeg": "X-ray showing healthy lungs.",
    "ct1.jpeg": "CT scan with small lesion in left lung.",
    "ct2.jpeg": "CT scan showing normal lung tissue.",
    "mri1.png": "MRI scan showing normal brain anatomy.",
    "mri2.jpg": "MRI scan showing mild temporal lobe swelling."
}

data_entries = [{"image": f"data/images/{img}", "caption": captions[img]} for img in ALL_IMAGES]

train, val = train_test_split(data_entries, test_size=0.2, random_state=42)

os.makedirs("./dataset", exist_ok=True)
with open("./dataset/train.json", "w") as f:
    json.dump(train, f, indent=4)
with open("./dataset/val.json", "w") as f:
    json.dump(val, f, indent=4)

print("✅ train.json and val.json created in ./dataset/")
