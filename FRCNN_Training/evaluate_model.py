import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import csv
import glob

from dataset import RoboflowCocoDataset
from transforms import Compose, ToTensor, Resize, collate_fn
from utils import evaluate_coco

def evaluate_checkpoint(checkpoint_path, val_loader, val_dataset, device):
    """Evaluate a specific checkpoint"""
    
    # Load model
    num_classes = 1 + len(val_dataset.cat_id_map)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Get training metrics from checkpoint
    epoch = checkpoint.get('epoch', 0)
    train_loss = checkpoint.get('train_loss', 0)
    val_loss = checkpoint.get('val_loss', 0)
    
    # Evaluate
    print(f"Evaluating epoch {epoch}...")
    coco_stats = evaluate_coco(model, val_loader, val_dataset.coco, device, val_dataset.inv_cat_id_map)
    
    if coco_stats:
        map5095 = coco_stats["mAP@[.5:.95]"]
        map50 = coco_stats["mAP@0.50"]
        precision = coco_stats.get("Precision", 0)
        recall = coco_stats.get("Recall", 0)
    else:
        map5095, map50, precision, recall = 0, 0, 0, 0
    
    return {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'mAP50': map50,
        'mAP@[.5:.95]': map5095,
        'Precision': precision,
        'Recall': recall
    }

def main():
    # Setup paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(SCRIPT_DIR, "Near_Drowning_Detector-7")
    VAL_IMAGES = os.path.join(DATASET_DIR, "valid")
    VAL_ANN = os.path.join(DATASET_DIR, "valid/annotate", "fixed__annotations.coco.json")
    
    # Load validation dataset
    val_dataset = RoboflowCocoDataset(VAL_IMAGES, VAL_ANN, transforms=Compose([Resize((640, 640)), ToTensor()]))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    # Find all checkpoints
    checkpoint_pattern = "./checkpoints/fasterrcnn_epoch*.pth"
    checkpoint_files = sorted(glob.glob(checkpoint_pattern), key=lambda x: int(x.split('epoch')[1].split('.')[0]))
    
    if not checkpoint_files:
        print("No checkpoint files found!")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoints to evaluate")
    
    # Evaluate all checkpoints
    results = []
    for ckpt_path in checkpoint_files:
        try:
            result = evaluate_checkpoint(ckpt_path, val_loader, val_dataset, device)
            results.append(result)
            print(f"Epoch {result['epoch']}: mAP50={result['mAP50']:.4f}, mAP@[.5:.95]={result['mAP@[.5:.95]']:.4f}")
        except Exception as e:
            print(f"Error evaluating {ckpt_path}: {e}")
    
    # Save results to CSV
    csv_path = "./evaluation_results.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "mAP50", "mAP@[.5:.95]", "Precision", "Recall"])
        
        for result in results:
            writer.writerow([
                result['epoch'],
                result['train_loss'],
                result['val_loss'],
                result['mAP50'],
                result['mAP@[.5:.95]'],
                result['Precision'],
                result['Recall']
            ])
    
    print(f"\n📊 Evaluation complete! Results saved to: {csv_path}")
    
    # Print summary
    if results:
        best_map50 = max(results, key=lambda x: x['mAP50'])
        best_map5095 = max(results, key=lambda x: x['mAP@[.5:.95]'])
        
        print(f"\n🏆 Best mAP50: {best_map50['mAP50']:.4f} at epoch {best_map50['epoch']}")
        print(f"🏆 Best mAP@[.5:.95]: {best_map5095['mAP@[.5:.95]']:.4f} at epoch {best_map5095['epoch']}")

if __name__ == "__main__":
    main()