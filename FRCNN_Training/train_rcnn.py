import os
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from dataset import RoboflowCocoDataset
from transforms import Compose, ToTensor, RandomHorizontalFlip, Resize, collate_fn
from utils import plot_training_curves

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "Near_Drowning_Detector-7")

# Define paths - Use the already fixed annotation files
TRAIN_IMAGES = os.path.join(DATASET_DIR, "train")
VAL_IMAGES = os.path.join(DATASET_DIR, "valid")

# Use the fixed annotation files created by fix_annotation.py
TRAIN_ANN = os.path.join(DATASET_DIR, "train/annotate", "fixed__annotations.coco.json")
VAL_ANN = os.path.join(DATASET_DIR, "valid/annotate", "fixed__annotations.coco.json")

def get_transform(train):
    transforms = [Resize((640, 640)), ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

def main():
    # --- Dataset & Dataloaders ---
    print(f"Using training annotations: {TRAIN_ANN}")
    print(f"Using validation annotations: {VAL_ANN}")
    
    train_dataset = RoboflowCocoDataset(TRAIN_IMAGES, TRAIN_ANN, transforms=get_transform(train=True))
    val_dataset = RoboflowCocoDataset(VAL_IMAGES, VAL_ANN, transforms=get_transform(train=False))

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # --- Model ---
    num_classes = 1 + len(train_dataset.cat_id_map)
    print(f"Number of classes: {num_classes}")
    print(f"Class labels: {train_dataset.labels}")
    
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

    # --- Training ---
    num_epochs = 40
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
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
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # --- Save checkpoint every epoch ---
        ckpt_path = f"./checkpoints/fasterrcnn_epoch{epoch+1}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        }, ckpt_path)

    # --- Plot training curves (loss only) ---
    plot_training_curves(train_losses, val_losses, [], [], [], [])
    print(f"\n📊 Training completed! Checkpoints saved in ./checkpoints/")
    print(f"📊 Run the evaluation script to generate CSV with mAP metrics.")

if __name__ == "__main__":
    main()