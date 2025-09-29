# CPU only
#pip install torch torchvision torchaudio
#pip install opencv-python

import torchvision
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import os

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load model with COCO weights (since training started from pretrained=True)
weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1

# Define the model (e.g., Faster R-CNN with ResNet-50 backbone)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=None,
    weights_backbone=None
)

num_classes = 6  # background + your class(es)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model_path = "../Near_Drowning_Repository/Model/Faster_RCNN/FRCNN_2.pth"
video_path = "../Near_Drowning_Repository/test_video/test2.mp4"
output_dir = "../Near_Drowning_Repository/output_vid/FRCNN/"

video_name = os.path.basename(video_path)

# Load your trained weights
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()


cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = os.path.join(output_dir, f"RCNN_{video_name}")
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor
    img = torch.from_numpy(frame).permute(2,0,1).float() / 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        predictions = model(img)[0]

    # Draw boxes
    for box, score in zip(predictions["boxes"], predictions["scores"]):
        if score > 0.5:
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()