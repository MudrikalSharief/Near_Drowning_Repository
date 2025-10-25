import os
import json
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from dataset import RoboflowCocoDataset
from transforms import Compose, ToTensor, Resize, collate_fn

def get_transform():
    return Compose([Resize((640, 640)), ToTensor()])

def generate_predictions():
    # --- Paths ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(SCRIPT_DIR, "Near_Drowning_Detector-8")
    VAL_IMAGES = os.path.join(DATASET_DIR, "test")
    VAL_ANN = os.path.join(DATASET_DIR, "test", "_annotations.coco.json")

    CHECKPOINT_PATH = "../checkpoints/fasterrcnn_epoch40.pth"  # update to your file
    OUTPUT_JSON = "C:/Users/Lenovo/Marcelino-Portfolio/Near_Drowning_Repository/FRCNN_Training/fasterrcnn_preds.json"

    os.makedirs("results", exist_ok=True)

    # --- Load dataset ---
    val_dataset = RoboflowCocoDataset(VAL_IMAGES, VAL_ANN, transforms=get_transform())
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    # 🔧 num_workers=0 avoids multiprocessing issues on Windows

    # --- Model ---
    num_classes = 1 + len(val_dataset.cat_id_map)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Load checkpoint ---
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✅ Loaded checkpoint from {CHECKPOINT_PATH}")

    results = []
    print("\n🔍 Generating predictions on validation set...")

    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(val_loader)):
            images = [img.to(device) for img in images]
            outputs = model(images)

            output = outputs[0]
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            image_id = targets[0]["image_id"].item()

            for box, score, label in zip(boxes, scores, labels):
                if score < 0.3:  # optional threshold
                    continue
                x_min, y_min, x_max, y_max = box
                width, height = x_max - x_min, y_max - y_min

                results.append({
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": [float(x_min), float(y_min), float(width), float(height)],
                    "score": float(score)
                })

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f)

    print(f"\n📁 Saved COCO-style predictions to: {OUTPUT_JSON}")

# ---------------------------------------
# ✅ Main entry point (Windows requirement)
# ---------------------------------------
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # extra safety for Windows
    generate_predictions()
