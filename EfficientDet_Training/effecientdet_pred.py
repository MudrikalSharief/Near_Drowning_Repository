import os
import json
import torch
from tqdm import tqdm
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import postprocess, preprocess, invert_affine
from efficientdet.dataset import CocoDataset, Resizer, Normalizer
from torch.utils.data import DataLoader

# =============================
# 🧠 Configurations
# =============================
num_classes = 2  # 👈 CHANGE THIS to match your dataset
compound_coef = 3  # same as training (-c 3)
project_name = "Near_Drowning_Detector"
weight_path = "C:/Users/Lenovo/Marcelino-Portfolio/Near_Drowning_Repository/Yet-Another-EfficientDet-Pytorch/efficientdet-d3_20_111500.pth"
output_json = "C:/Users/Lenovo/Marcelino-Portfolio/Near_Drowning_Repository/Yet-Another-EfficientDet-Pytorch/efficientdet_preds.json"
data_dir = "C:/Users/Lenovo/Marcelino-Portfolio/Near_Drowning_Repository/Yet-Another-EfficientDet-Pytorch/datasets/Near_Drowning_Detector/val2017"  # adjust to your val path
val_dataset = CocoDataset(
    root_dir=r"C:\Users\Lenovo\Marcelino-Portfolio\Near_Drowning_Repository\Yet-Another-EfficientDet-Pytorch\datasets\Near_Drowning_Detector",
    set='val2017',
    transform=None
)


os.makedirs("results", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# 🧩 Load model
# =============================
print(f"Loading EfficientDet (D{compound_coef}) from {weight_path}")

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=num_classes)
model.load_state_dict(torch.load(weight_path))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# =============================
# 🧩 Load dataset
# =============================
from utils.utils import get_last_weights
from torchvision import transforms as T
from efficientdet.dataset import CocoDataset, collater

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collater)

# =============================
# 🧩 Generate predictions   
# =============================
results = []
print("🔍 Generating predictions...")

for idx, data in enumerate(tqdm(val_loader)):
    imgs = data["img"]
    scales = data.get("scale", [1.0])  # ✅ prevent KeyError
    img_ids = data.get("img_id", [idx])  # fallback if not provided

    imgs = imgs.to(device).float()
    with torch.no_grad():
        features, regression, classification, anchors = model(imgs)
        preds = postprocess(imgs, anchors, regression, classification,
                    BBoxTransform(), ClipBoxes(),
                    0.3, 0.5)


   # preds = invert_affine(scales, preds)

    for i, pred in enumerate(preds):
        if pred is None:
            continue
        image_id = int(img_ids[i])
        boxes = pred["rois"]
        scores = pred["scores"]
        labels = pred["class_ids"]

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            results.append({
                "image_id": image_id,
                "category_id": int(label + 1),  # shift to match COCO IDs
                "bbox": [float(x1), float(y1), float(width), float(height)],
                "score": float(score)
            })

# =============================
# 🧩 Save to JSON
# =============================
with open(output_json, "w") as f:
    json.dump(results, f)
print(f"\n📁 Saved predictions to {output_json}")
