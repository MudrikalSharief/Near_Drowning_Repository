import torchvision
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import os

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Wait for user to press any key before running
input("Press Enter to continue...")

# ✅ FIXED: Load model with pretrained backbone
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1  # Use pretrained weights as base
)

# ✅ FIXED: Change from 6 to 5 classes to match your saved model
num_classes = 5  # This matches your saved model (background + 4 object classes)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# ✅ FIXED: Update paths to be relative to current working directory
model_path = "Model/Faster_RCNN/fasterrcnn_epoch29.pth"
video_path = "test_video/swimming13.mp4"
output_dir = "output_vid/FRCNN/"

video_name = os.path.basename(video_path)

# ✅ FIXED: Handle the checkpoint format properly
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

# Check if it's a full checkpoint or just state_dict
if 'model_state_dict' in checkpoint:
    # It's a full checkpoint with metadata
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
else:
    # It's just the state_dict
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = os.path.join(output_dir, f"RCNN_{video_name}")
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# ✅ Define class names and colors for each class
class_info = {
    0: {'name': 'background', 'color': (128, 128, 128)},      # Gray
    1: {'name': 'Lunod', 'color': (0, 0, 255)},       # Red (BGR format)
    2: {'name': 'near_drowning', 'color': (255, 0, 255)},   # Magenta
    3: {'name': 'Out_of_water', 'color': (0, 255, 0)},        # Green
    4: {'name': 'swimming', 'color': (255, 255, 0)}           # Cyan
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor and move to device
    img = torch.from_numpy(frame).permute(2,0,1).float() / 255.0
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img)[0]

    # Move predictions back to CPU for drawing
    boxes = predictions["boxes"].cpu()
    scores = predictions["scores"].cpu()
    labels = predictions["labels"].cpu()

    # Draw boxes with different colors for each class
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = box.int().tolist()
            
            # Get class info (name and color)
            label_id = label.item()
            class_name = class_info.get(label_id, {}).get('name', f'Class {label_id}')
            color = class_info.get(label_id, {}).get('color', (255, 255, 255))  # Default white
            
            # Draw bounding box with class-specific color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Create background rectangle for text for better visibility
            text = f'{class_name}: {score:.2f}'
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background rectangle for text
            cv2.rectangle(frame, (x1, y1-text_height-10), (x1+text_width, y1), color, -1)
            
            # Add label text in white color for contrast
            cv2.putText(frame, text, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(frame)
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()