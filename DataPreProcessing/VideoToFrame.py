import cv2
import os

input_vid_name = 'C:/Users/user/TurtleNet/test2.mp4'
output_frame_folder = 'test2'

cap = cv2.VideoCapture(input_vid_name)

vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if not os.path.exists(output_frame_folder):
  os.makedirs(output_frame_folder)

for framenum in range(0, vid_length):

    print(framenum)
    cap.set(cv2.CAP_PROP_FRAME_COUNT, framenum)
    ret, frame = cap.read()

    if ret is False:
        break

    # Image Processing
    cv2.imwrite(output_frame_folder + '/' + "frame_2_"+str(framenum) + '.jpg', frame)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:  # Escape (ESC)
        break

cap.release()
cv2.destroyAllWindows()