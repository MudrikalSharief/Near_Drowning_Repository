import os
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import csv
import glob

from dataset import RoboflowCocoDataset
from transforms import Compose, ToTensor, RandomHorizontalFlip, Resize, collate_fn
from utils import evaluate_coco, plot_training_curves

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "Near_Drowning_Detector-7")

TRAIN_IMAGES = os.path.join(DATASET_DIR, "train")
VAL_IMAGES = os.path.join(DATASET_DIR, "valid")
TRAIN_ANN = os.path.join(DATASET_DIR, "train/annotate", "_annotations.coco.json")
VAL_ANN = os.path.join(DATASET_DIR, "valid/annotate", "_annotations.coco.json")

def get_transform(train):
    transforms = [Resize((640, 640)), ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


def main():
    # --- Dataset & Dataloaders ---
    train_dataset = RoboflowCocoDataset(TRAIN_IMAGES, TRAIN_ANN, transforms=get_transform(train=True))
    val_dataset = RoboflowCocoDataset(VAL_IMAGES, VAL_ANN, transforms=get_transform(train=False))

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # --- Model ---
    num_classes = 1 + len(train_dataset.cat_id_map)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)
    model.to(device)

    # --- Optimizer & Scheduler ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # --- Resume from checkpoint if available ---
    ckpt_files = glob.glob("./checkpoints/fasterrcnn_epoch*.pth")
    if ckpt_files:
        latest_ckpt = max(ckpt_files, key=os.path.getctime)
        checkpoint = torch.load(latest_ckpt)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
        print("No checkpoint found, starting from scratch.")

    # --- Training ---
    num_epochs = 40
    train_losses, val_losses, map50s, map5095s, precisions, recalls = [], [], [], [], [], []

    csv_path = "./training_log.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "mAP50", "mAP@[.5:.95]", "Precision", "Recall"])

    coco_gt = val_dataset.coco

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for images, targets in loop:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            loop.set_postfix(loss=losses.item())

        lr_scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation loss ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                model.train()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                model.eval()

                val_loss += losses.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Validation loss: {avg_val_loss:.4f}")

        # --- COCO evaluation ---
        print(f"\nRunning COCO evaluation after epoch {epoch+1}...")
        coco_stats = evaluate_coco(model, val_loader, val_dataset.coco, device, val_dataset.inv_cat_id_map)
        if coco_stats:
            map5095 = coco_stats["mAP@[.5:.95]"]
            map50 = coco_stats["mAP@0.50"]
            precision = coco_stats.get("Precision", 0)
            recall = coco_stats.get("Recall", 0)
        else:
            map5095, map50, precision, recall = 0, 0, 0, 0

        map5095s.append(map5095)
        map50s.append(map50)
        precisions.append(precision)
        recalls.append(recall)

        print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, "
              f"mAP50={map50:.4f}, mAP@[.5:.95]={map5095:.4f}, "
              f"Precision={precision:.4f}, Recall={recall:.4f}")

        # --- Save results ---
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss, map50, map5095, precision, recall])

        ckpt_path = f"./checkpoints/fasterrcnn_epoch{epoch+1}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)

    # --- Plot curves ---
    plot_training_curves(train_losses, val_losses, map50s, map5095s, precisions, recalls)
    print(f"\n📊 Training log saved at: {csv_path}")


if __name__ == "__main__":
    main()
