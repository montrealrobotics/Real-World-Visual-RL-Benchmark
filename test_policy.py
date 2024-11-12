import cv2
import numpy as np

fps = 30

camera = cv2.VideoCapture(4)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
camera.set(cv2.CAP_PROP_FPS, fps)
codec = cv2.VideoWriter_fourcc("M", "J", "P", "G")
camera.set(cv2.CAP_PROP_FOURCC, codec)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

background_img = cv2.imread("assets/imgs/img1.jpg")

frames = []
results = []

while True:
    return_val, frame = camera.read()
    print(frame.shape)
    frames.append(frame)

    if not return_val:
        break

    background_img = cv2.resize(background_img, (frame.shape[1], frame.shape[0]))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([32, 25, 67])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    mask_inv = cv2.bitwise_not(mask)

    not_green_stuff = cv2.bitwise_and(frame, frame, mask=mask_inv)
    green_stuff = cv2.bitwise_and(background_img, background_img, mask=mask)

    result = cv2.add(not_green_stuff, green_stuff)

    results.append(result)

    if len(frames) == 60:
        break
camera.release()

output_file = "output_video.avi"
frame_height, frame_width, _ = frames[0].shape
video_writer = cv2.VideoWriter(output_file, codec, fps, (frame_width, frame_height))

for frame in results:
    video_writer.write(frame)

video_writer.release()
