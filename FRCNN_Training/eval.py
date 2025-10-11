import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import glob
import csv

from dataset import RoboflowCocoDataset
from transforms import Compose, ToTensor, Resize, collate_fn
from utils import evaluate_coco

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "Near_Drowning_Detector-7")

VAL_IMAGES = os.path.join(DATASET_DIR, "valid")
VAL_ANN = os.path.join(DATASET_DIR, "valid/annotate", "_annotations.coco.json")

def get_transform():
    return Compose([Resize((640, 640)), ToTensor()])

def main():
    val_dataset = RoboflowCocoDataset(VAL_IMAGES, VAL_ANN, transforms=get_transform())
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    num_classes = 1 + len(val_dataset.cat_id_map)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    checkpoint_dir = "./checkpoints"
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "fasterrcnn_epoch*.pth")))

    if not ckpt_files:
        print("No checkpoints found in ./checkpoints")
        return

    csv_path = "./checkpoint_evaluation.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint", "mAP@[.5:.95]", "mAP@0.50", "Precision", "Recall"])

        for ckpt_path in ckpt_files:
            print(f"\nEvaluating checkpoint: {ckpt_path}")
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            model.to(device)

            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

            coco_stats = evaluate_coco(model, val_loader, val_dataset.coco, device, val_dataset.inv_cat_id_map)
            if coco_stats:
                print(f"mAP@[.5:.95]: {coco_stats['mAP@[.5:.95]']:.4f}, "
                      f"mAP@0.50: {coco_stats['mAP@0.50']:.4f}, "
                      f"Precision: {coco_stats['Precision']:.4f}, "
                      f"Recall: {coco_stats['Recall']:.4f}")
                writer.writerow([
                    os.path.basename(ckpt_path),
                    coco_stats['mAP@[.5:.95]'],
                    coco_stats['mAP@0.50'],
                    coco_stats['Precision'],
                    coco_stats['Recall']
                ])
            else:
                print("No predictions to evaluate.")
                writer.writerow([os.path.basename(ckpt_path), 0, 0, 0, 0])

    print(f"\n📊 Evaluation results saved at: {csv_path}")

if __name__ == "__main__":
    main()