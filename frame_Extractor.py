import cv2
import os

vid = cv2.VideoCapture('C:/Users/Lenovo Slim i3/Documents/GitHub/Near_Drowning_Repository/videos/video1.mp4')
currentframe = 0

if not os.path.exists('frame_extracted'):
    os.makedirs('frame_extracted')

while(True):
    success, frame = vid.read()

    cv2.imshow("Output", frame)
    cv2.imwrite('./frame_extracted' + str(currentframe) + '.jpg', frame)
    currentframe += 1

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()