# if only use CPU
# pip install torch torchvision torchaudio
# Yif use GPU
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
import cv2
import torch
from ultralytics import YOLO
import os 

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Wait for user to press any key before running
input("Press Enter to continue...")

# Load the YOLO model and move it to the device
model = YOLO("../Near_Drowning_Repository/Model/yolov8/latest_model.pt")
model.to(device)

# Open the video file
video_path = "../Near_Drowning_Repository/test_video/swimming13.mp4"
cap = cv2.VideoCapture(video_path)
video_name = os.path.basename(video_path)
print("Video name:", video_name)

# Optional: set output video writer if you want to save results
save_output = True
if save_output:
    output_dir = '../Near_Drowning_Repository/output_vid/yolov8/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = os.path.join(output_dir, f"yolo_{video_name}")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video or cannot read the frame.")
        break

    # Run YOLO inference on the frame
    results = model(frame, device=device)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Show the annotated frame
    cv2.imshow("YOLO Inference", annotated_frame)

    # Save to output file if enabled
    if save_output:
        out.write(annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything
cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
