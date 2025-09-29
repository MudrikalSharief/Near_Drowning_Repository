# pip install lap

from ultralytics import YOLO
import os

# Load the YOLO model
model = YOLO("../Near_Drowning_Repository/Model/yolov8/PD_default.pt")

# Input video
video_path = "../Near_Drowning_Repository/test_video/test.mp4"
video_name = os.path.basename(video_path)
print("Video name:", video_name)

# Output directory
output_dir = "../Near_Drowning_Repository/output_vid/yolov8/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, f"tracked_{video_name}")

# Run tracking
results = model.track(
    source=video_path,          # your video
    tracker="bytetrack.yaml",   # or "strongsort.yaml"
    show=True,                  # display window with IDs
    save=True,                  # save annotated video
    project=output_dir,         # save folder
    name=f"yolo_{video_name}"   # output subfolder name
)

print(f"Tracking complete. Saved at: {output_dir}")
