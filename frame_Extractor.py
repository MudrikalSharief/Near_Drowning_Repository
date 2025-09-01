import cv2
import os

video_path = "C:/Users/Lenovo Slim i3/Documents/GitHub/Near_Drowning_Repository/videos/video1.mp4"
output_dir = "C:/Users/Lenovo Slim i3/Documents/GitHub/Near_Drowning_Repository/frame_extracted"

vid = cv2.VideoCapture(video_path)
currentframe = 0

if not os.path.exists('frame_extracted'):
    os.makedirs('frame_extracted')


while(True):
    success, frame = vid.read()

    if not success:  # stop if no more frames
        print("No more frames or cannot read the video.")
        break

    cv2.imshow("Output", frame)

    frame_filename = os.path.join(output_dir, f"frame_{currentframe}.jpg")
    cv2.imwrite(frame_filename, frame)
    currentframe += 1

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()