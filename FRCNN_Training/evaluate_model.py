import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import csv
import glob
import json
from sklearn.metrics import f1_score, accuracy_score 
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Assuming these custom modules are available in your laptop environment
from dataset import RoboflowCocoDataset
from transforms import Compose, ToTensor, Resize, collate_fn

# =============================
# 📊 Comprehensive COCO Evaluation (Same as EfficientDet)
# =============================
def evaluate_coco_comprehensive(model, data_loader, coco_gt, device, confidence_threshold=0.05):
    """
    Comprehensive COCO evaluation matching EfficientDet metrics
    """
    model.eval()
    results = []
    
    print("🔍 Generating predictions for COCO evaluation...")
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = list(img.to(device) for img in images)
            
            # Get predictions
            outputs = model(images)
            
            # Process each image in the batch
            for img_idx, output in enumerate(outputs):
                # Get image info
                image_id = targets[img_idx]['image_id'].item()
                
                # Filter predictions by confidence
                scores = output['scores'].cpu().numpy()
                boxes = output['boxes'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                
                # Apply confidence threshold
                valid_indices = scores >= confidence_threshold
                scores = scores[valid_indices]
                boxes = boxes[valid_indices]
                labels = labels[valid_indices]
                
                # Convert to COCO format
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Ensure positive width and height
                    if width > 0 and height > 0:
                        results.append({
                            "image_id": int(image_id),
                            "category_id": int(label),  # Faster R-CNN uses 1-based labels
                            "bbox": [float(x1), float(y1), float(width), float(height)],
                            "score": float(score)
                        })
    
    if not results:
        print("⚠️ No predictions generated!")
        return None
    
    # Save predictions to temporary file
    pred_file = "temp_frcnn_predictions.json"
    with open(pred_file, 'w') as f:
        json.dump(results, f)
    
    try:
        # Load predictions and run COCO evaluation
        coco_dt = coco_gt.loadRes(pred_file)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics (same as EfficientDet)
        metrics = {
            "mAP@0.50:0.95": float(coco_eval.stats[0]),
            "mAP@0.50": float(coco_eval.stats[1]),
            "mAP@0.75": float(coco_eval.stats[2]),
            "mAP_small": float(coco_eval.stats[3]),
            "mAP_medium": float(coco_eval.stats[4]),
            "mAP_large": float(coco_eval.stats[5]),
            "AR@0.5:0.95": float(coco_eval.stats[8]),
            "num_predictions": len(results)
        }
        
        # Calculate F1-Score (same formula as EfficientDet)
        precision = metrics["mAP@0.50"]
        recall = metrics["AR@0.5:0.95"]
        
        if precision + recall > 0:
            f1_score_val = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score_val = 0.0
            
        metrics["Precision"] = precision
        metrics["Recall"] = recall
        metrics["F1_Score"] = f1_score_val
        
        # Clean up temp file
        os.remove(pred_file)
        
        return metrics
        
    except Exception as e:
        print(f"❌ COCO evaluation failed: {e}")
        # Clean up temp file
        if os.path.exists(pred_file):
            os.remove(pred_file)
        return None

# =============================
# 📊 Image-level Classification Metrics
# =============================
def calculate_classification_metrics(model, data_loader, device, confidence_threshold=0.5):
    """
    Image-level binary classification (Object Present vs. No Object)
    """
    y_true = []
    y_pred = []
    model.eval()

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            
            # Ground Truth: Check if image has any objects
            target_dict = targets[0]
            has_gt_objects = len(target_dict["boxes"]) > 0
            y_true.append(1 if has_gt_objects else 0)
            
            # Prediction: Check if model detects any objects above threshold
            outputs = model(images)
            has_detection = False
            
            if len(outputs[0]["scores"]) > 0:
                high_confidence_scores = outputs[0]["scores"] >= confidence_threshold
                has_detection = high_confidence_scores.any().item()
            
            y_pred.append(1 if has_detection else 0)
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    return f1, accuracy

# =============================
# 📊 Comprehensive Checkpoint Evaluation
# =============================
def evaluate_checkpoint(checkpoint_path, val_loader, val_dataset, device):
    """Evaluate a specific checkpoint with comprehensive metrics"""
    
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
    
    # Get training info
    epoch = checkpoint.get('epoch', 0)
    train_loss = checkpoint.get('train_loss', 0)
    val_loss = checkpoint.get('val_loss', 0)
    
    print(f"\n🔍 Evaluating Epoch {epoch}...")
    print("-" * 50)
    
    # 1. Comprehensive COCO Evaluation
    print("📊 Running COCO evaluation...")
    coco_metrics = evaluate_coco_comprehensive(model, val_loader, val_dataset.coco, device)
    
    # 2. Image-level Classification Metrics
    print("📊 Running classification evaluation...")
    try:
        cls_f1, cls_accuracy = calculate_classification_metrics(model, val_loader, device)
    except Exception as e:
        print(f"⚠️ Classification metrics failed: {e}")
        cls_f1, cls_accuracy = 0.0, 0.0
    
    # Combine all metrics
    result = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'num_predictions': 0,
        # COCO Detection Metrics (same as EfficientDet)
        'mAP@0.50:0.95': 0.0,
        'mAP@0.50': 0.0,
        'mAP@0.75': 0.0,
        'mAP_small': 0.0,
        'mAP_medium': 0.0,
        'mAP_large': 0.0,
        'AR@0.5:0.95': 0.0,
        'Detection_Precision': 0.0,
        'Detection_Recall': 0.0,
        'Detection_F1_Score': 0.0,
        # Image Classification Metrics
        'Classification_F1_Score': cls_f1,
        'Classification_Accuracy': cls_accuracy
    }
    
    # Update with COCO metrics if available
    if coco_metrics:
        result.update({
            'mAP@0.50:0.95': coco_metrics['mAP@0.50:0.95'],
            'mAP@0.50': coco_metrics['mAP@0.50'],
            'mAP@0.75': coco_metrics['mAP@0.75'],
            'mAP_small': coco_metrics['mAP_small'],
            'mAP_medium': coco_metrics['mAP_medium'],
            'mAP_large': coco_metrics['mAP_large'],
            'AR@0.5:0.95': coco_metrics['AR@0.5:0.95'],
            'Detection_Precision': coco_metrics['Precision'],
            'Detection_Recall': coco_metrics['Recall'],
            'Detection_F1_Score': coco_metrics['F1_Score'],
            'num_predictions': coco_metrics['num_predictions']
        })
    
    # Print summary
    print(f"✅ Epoch {epoch} Results:")
    print(f"   📊 mAP@0.50:0.95: {result['mAP@0.50:0.95']:.4f}")
    print(f"   📊 mAP@0.50:     {result['mAP@0.50']:.4f}")
    print(f"   📊 Detection F1:  {result['Detection_F1_Score']:.4f}")
    print(f"   📊 Class F1:      {result['Classification_F1_Score']:.4f}")
    print(f"   📊 Predictions:   {result['num_predictions']}")
    
    return result

# =============================
# 📊 Main Evaluation Function
# =============================
def main():
    # Setup paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(SCRIPT_DIR, "Near_Drowning_Detector-8")
    
    # Use validation set (same as EfficientDet)
    VAL_IMAGES = os.path.join(DATASET_DIR, "valid")
    VAL_ANN = os.path.join(DATASET_DIR, "valid/annotate", "_annotations.coco.json")
    
    # Load validation dataset
    val_dataset = RoboflowCocoDataset(VAL_IMAGES, VAL_ANN, transforms=Compose([Resize((640, 640)), ToTensor()]))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"🚀 Using device: {device}")
    print(f"📁 Validation images: {len(val_dataset)}")
    
    # Find all checkpoints
    checkpoint_pattern = "../checkpoints/fasterrcnn_epoch*.pth"
    checkpoint_files = sorted(glob.glob(checkpoint_pattern), key=lambda x: int(x.split('epoch')[1].split('.')[0]))
    
    if not checkpoint_files:
        print("❌ No checkpoint files found!")
        return
    
    print(f"📁 Found {len(checkpoint_files)} checkpoints to evaluate")
    
    # Evaluate all checkpoints
    results = []
    for ckpt_path in checkpoint_files:
        try:
            result = evaluate_checkpoint(ckpt_path, val_loader, val_dataset, device)
            results.append(result)
        except Exception as e:
            print(f"❌ Error evaluating {ckpt_path}: {e}")
    
    # Save comprehensive results to CSV
    csv_path = "./comprehensive_evaluation_results.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "num_predictions",
            "mAP@0.50:0.95", "mAP@0.50", "mAP@0.75", 
            "mAP_small", "mAP_medium", "mAP_large",
            "AR@0.5:0.95", "Detection_Precision", "Detection_Recall", "Detection_F1_Score",
            "Classification_F1_Score", "Classification_Accuracy"
        ])
        
        for result in results:
            writer.writerow([
                result['epoch'], result['train_loss'], result['val_loss'], result['num_predictions'],
                result['mAP@0.50:0.95'], result['mAP@0.50'], result['mAP@0.75'],
                result['mAP_small'], result['mAP_medium'], result['mAP_large'],
                result['AR@0.5:0.95'], result['Detection_Precision'], result['Detection_Recall'], result['Detection_F1_Score'],
                result['Classification_F1_Score'], result['Classification_Accuracy']
            ])
    
    # Save detailed JSON results (same as EfficientDet)
    json_path = "./frcnn_evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            "model_info": {
                "model_type": "Faster R-CNN",
                "backbone": "ResNet50-FPN",
                "num_classes": len(val_dataset.cat_id_map) + 1,
                "dataset_size": len(val_dataset)
            },
            "results": results
        }, f, indent=4)
    
    print(f"\n📊 Evaluation complete!")
    print(f"💾 CSV results saved to: {csv_path}")
    print(f"💾 JSON results saved to: {json_path}")
    
    # Print summary (same format as EfficientDet)
    if results:
        best_map50 = max(results, key=lambda x: x['mAP@0.50'])
        best_det_f1 = max(results, key=lambda x: x['Detection_F1_Score'])
        best_cls_f1 = max(results, key=lambda x: x['Classification_F1_Score'])
        
        print(f"\n" + "="*60)
        print("🏆 BEST RESULTS SUMMARY")
        print("="*60)
        print(f"🥇 Best mAP@0.50:        {best_map50['mAP@0.50']:.4f} (Epoch {best_map50['epoch']})")
        print(f"🥇 Best Detection F1:    {best_det_f1['Detection_F1_Score']:.4f} (Epoch {best_det_f1['epoch']})")
        print(f"🥇 Best Classification F1: {best_cls_f1['Classification_F1_Score']:.4f} (Epoch {best_cls_f1['epoch']})")
        print("="*60)

if __name__ == "__main__":
    main()