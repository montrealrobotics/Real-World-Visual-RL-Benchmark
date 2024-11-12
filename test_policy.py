import cv2
import numpy as np

fps = 30
img_height = 256
img_width = 256

camera = cv2.VideoCapture(4)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
camera.set(cv2.CAP_PROP_FPS, fps)
codec = cv2.VideoWriter_fourcc("M", "J", "P", "G")
camera.set(cv2.CAP_PROP_FOURCC, codec)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

distraction_img = cv2.imread("assets/imgs/img1.jpg")
distraction_img = cv2.resize(distraction_img, (img_width, img_height))

frames = []

lower_green = np.array([32, 25, 67])
upper_green = np.array([90, 255, 255])

while True:
    return_val, frame = camera.read()

    if not return_val:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower_green, upper_green)

    mask_inv = cv2.bitwise_not(mask)
    frame_without_mask = cv2.bitwise_and(frame, frame, mask=mask_inv)
    distraction_img_at_mask = cv2.bitwise_and(distraction_img, distraction_img, mask=mask)
    frame = cv2.add(frame_without_mask, distraction_img_at_mask)

    frames.append(frame)

    if len(frames) == 60:
        break
camera.release()

output_file = "output_video.avi"
video_writer = cv2.VideoWriter(output_file, codec, fps, (img_width, img_height))

for frame in frames:
    video_writer.write(frame)

video_writer.release()
