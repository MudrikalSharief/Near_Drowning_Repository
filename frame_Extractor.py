import cv2
import os

video_path = "C:/Users/Lenovo Slim i3/Documents/GitHub/Near_Drowning_Repository/videos/bom_video_3.mp4"
output_dir = "C:/Users/Lenovo Slim i3/Documents/GitHub/Near_Drowning_Repository/frame_extracted"

video_name = os.path.basename(video_path)
print("Video name:", video_name)

vid = cv2.VideoCapture(video_path)
fps = vid.get(cv2.CAP_PROP_FPS)  # Get the video's FPS
frames_per_second_to_save = 5    # Change this to how many frames you want per second

currentframe = 0
savedframe = 0

if not os.path.exists('frame_extracted'):
    os.makedirs('frame_extracted')

video_name_no_ext = os.path.splitext(video_name)[0]
save_dir = os.path.join(output_dir, video_name_no_ext)
counter = 1

while os.path.exists(save_dir):
    save_dir = os.path.join(output_dir, f"{video_name_no_ext}_{counter}")
    counter += 1

os.makedirs(save_dir)
print(f"Frames will be saved to: {save_dir}")

while(True):
    success, frame = vid.read()

    if not success:  # stop if no more frames
        print("No more frames or cannot read the video.")
        break

    cv2.imshow("Output", frame)

    # Save only the desired frames per second
    if int(currentframe % (fps // frames_per_second_to_save)) == 0:
        frame_filename = os.path.join(save_dir, f"frame_{savedframe}.jpg")
        cv2.imwrite(frame_filename, frame)
        savedframe += 1
    currentframe += 1
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        print("Exiting on user command.")
        print("Total frames saved:", savedframe)
        break

print("Total frames saved:", savedframe)
vid.release()
cv2.destroyAllWindows()