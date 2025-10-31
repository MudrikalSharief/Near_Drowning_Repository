import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import csv
import glob
# Assuming scikit-learn is now installed
from sklearn.metrics import f1_score, accuracy_score 
import numpy as np

# Assuming these custom modules are available in your laptop environment
from dataset import RoboflowCocoDataset
from transforms import Compose, ToTensor, Resize, collate_fn
from utils import evaluate_coco

# --- NEW FUNCTION FOR IMAGE-LEVEL CLASSIFICATION METRICS ---
def calculate_classification_metrics(model, data_loader, device, confidence_threshold=0.5):
    """
    Performs image-level binary classification (Object Present vs. No Object).
    """
    y_true = []
    y_pred = []
    model.eval()

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            # This part is complex due to your custom collate_fn returning a tuple of tuples/lists.
            # We must correctly extract the image tensor and the target dict.
            # Based on standard PyTorch Detection utilities:
            images = list(img.to(device) for img in images) 
            
            # targets is a list of dictionaries (one per image in the batch).
            # Since batch_size=1, targets is a list containing one dictionary.
            target_dict = targets[0] 

            # 1. Determine Ground Truth (y_true)
            # If the image has ANY ground truth boxes, it's a Positive (1) image.
            is_positive_image = 1 if len(target_dict["boxes"]) > 0 else 0
            y_true.append(is_positive_image)
            
            # 2. Determine Prediction (y_pred)
            outputs = model(images)
            
            # Check if any detection has a score above the threshold
            has_detection = False
            if len(outputs[0]["scores"]) > 0:
                high_confidence_scores = outputs[0]["scores"] >= confidence_threshold
                if high_confidence_scores.any():
                    has_detection = True
            
            is_positive_prediction = 1 if has_detection else 0
            y_pred.append(is_positive_prediction)
            
    # Calculate F1-score for the positive class (Object Present = 1)
    f1 = f1_score(y_true, y_pred, labels=[1], average='binary', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    return f1, accuracy
# -------------------------------------------------------------

def evaluate_checkpoint(checkpoint_path, val_loader, val_dataset, device):
    """Evaluate a specific checkpoint"""
    
    # Load model
    num_classes = 1 + len(val_dataset.cat_id_map)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
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
    
    # Evaluate COCO mAP
    print(f"Evaluating epoch {epoch} (mAP)...")
    coco_stats = evaluate_coco(model, val_loader, val_dataset.coco, device, val_dataset.inv_cat_id_map) 
    
    # Initialize values to 0 in case the custom evaluate_coco fails to return them
    map5095, map50, precision, recall = 0, 0, 0, 0
    if coco_stats:
        map5095 = coco_stats.get("mAP@[.5:.95]", 0)
        map50 = coco_stats.get("mAP@0.50", 0)
        # Assuming your custom evaluate_coco returns these specific keys
        precision = coco_stats.get("Precision", 0)
        recall = coco_stats.get("Recall", 0)
    
    # --- CALCULATE CLASSIFICATION METRICS ---
    print(f"Evaluating epoch {epoch} (F1/Accuracy)...")
    
    # Use robust try-except to guarantee a return even if classification fails
    f1_score_cls, accuracy_cls = 0, 0
    try:
        f1_score_cls, accuracy_cls = calculate_classification_metrics(model, val_loader, device)
    except Exception as e:
        print(f"DEBUG: Classification metrics calculation failed: {e}")
        # The result will be 0, which prevents the main loop from crashing
    
    return {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'mAP50': map50,
        'mAP@[.5:.95]': map5095,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1_score_cls,  
        'Accuracy': accuracy_cls   
    }

def main():
    # Setup paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(SCRIPT_DIR, "Near_Drowning_Detector-8")
    
    # --- USING THE TEST SET FOR FINAL METRICS ---
    TEST_IMAGES = os.path.join(DATASET_DIR, "valid")
    TEST_ANN = os.path.join(DATASET_DIR, "valid/annotate", "_annotations.coco.json") 
    
    # Load TEST dataset
    # We rename to test_dataset/test_loader for clarity, but variable names don't break code
    test_dataset = RoboflowCocoDataset(TEST_IMAGES, TEST_ANN, transforms=Compose([Resize((640, 640)), ToTensor()]))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    # Find all checkpoints
    checkpoint_pattern = "../checkpoints/fasterrcnn_epoch*.pth"
    checkpoint_files = sorted(glob.glob(checkpoint_pattern), key=lambda x: int(x.split('epoch')[1].split('.')[0]))
    
    if not checkpoint_files:
        print("No checkpoint files found!")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoints to evaluate")
    
    # Evaluate all checkpoints
    results = []
    for ckpt_path in checkpoint_files:
        try:
            # Pass test_loader/test_dataset to the function
            result = evaluate_checkpoint(ckpt_path, test_loader, test_dataset, device)
            results.append(result)
            
            # --- FIXED PRINT LINE ---
            print(f"Epoch {result['epoch']}: mAP50={result['mAP50']:.4f}, F1-Score={result['F1_Score']:.4f}, Accuracy={result['Accuracy']:.4f}")
        except Exception as e:
            # We catch the error here and print a helpful message
            print(f"Error evaluating {ckpt_path}: {e}. Skipping epoch print.")
    
    # Save results to CSV
    csv_path = "./evaluation_results.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "mAP50", "mAP@[.5:.95]", "Precision", "Recall", "F1_Score", "Accuracy"])
        
        for result in results:
            writer.writerow([
                result['epoch'],
                result['train_loss'],
                result['val_loss'],
                result['mAP50'],
                result['mAP@[.5:.95]'],
                result['Precision'],
                result['Recall'],
                result['F1_Score'], 
                result['Accuracy'] 
            ])
    
    print(f"\n📊 Evaluation complete! Results saved to: {csv_path}")
    
    # Print summary
    if results:
        best_map50 = max(results, key=lambda x: x['mAP50'])
        best_f1 = max(results, key=lambda x: x['F1_Score'])
        
        print(f"\n🏆 Best mAP50: {best_map50['mAP50']:.4f} at epoch {best_map50['epoch']}")
        print(f"🏆 Best F1-Score: {best_f1['F1_Score']:.4f} at epoch {best_f1['epoch']}")

if __name__ == "__main__":
    main()