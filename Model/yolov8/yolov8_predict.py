import cv2
from ultralytics import YOLO
import os 
# Load the YOLO model
model = YOLO("C:/Users/Lenovo Slim i3/Documents/GitHub/Near_Drowning_Repository/Model/yolov8/PD_default.pt")

# Open the video file
video_path = "C:/Users/Lenovo Slim i3/Documents/GitHub/Near_Drowning_Repository/Model/yolov8/test_video/test2.mp4"
cap = cv2.VideoCapture(video_path)
video_name = os.path.basename(video_path)
print("Video name:", video_name)

# Optional: set output video writer if you want to save results
save_output = True
if save_output:
    output_dir = 'C:/Users/Lenovo Slim i3/Documents/GitHub/Near_Drowning_Repository/Model/yolov8/output_videos'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = os.path.join(output_dir, f"output_{video_name}")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video or cannot read the frame.")
        break

    # Run YOLO inference on the frame
    results = model(frame)

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
