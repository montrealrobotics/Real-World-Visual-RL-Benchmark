import random
import cv2
import numpy as np
from pathlib import Path

fps = 30
img_height = 720
img_width = 1280

camera = cv2.VideoCapture(4)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
codec = cv2.VideoWriter_fourcc("M", "J", "P", "G")
camera.set(cv2.CAP_PROP_FOURCC, codec)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

imgs_paths = [str(img_path) for img_path in Path("assets/imgs/davis").rglob("*.jpg")]

max_steps = 250

seed = 123
generator = random.Random(seed)

frames = []

lower_green = np.array([46, 8, 137])
upper_green = np.array([85, 255, 255])

step = 1
distraction_img = None
while step <= max_steps:
    return_val, frame = camera.read()

    if not return_val:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    rest_of_image = cv2.bitwise_and(frame, frame, mask=mask_inv)

    if distraction_img is None or step % 30 == 0:
        img_index = generator.randint(0, len(imgs_paths) - 1)
        distraction_img = cv2.imread(imgs_paths[img_index])
        distraction_img = cv2.resize(distraction_img, (img_width, img_height))

    mask_img = cv2.bitwise_and(distraction_img, distraction_img, mask=mask)
    frame = cv2.add(rest_of_image, mask_img)

    frames.append(frame)

    step += 1
camera.release()

output_file = "output_video.avi"
video_writer = cv2.VideoWriter(output_file, codec, fps, (img_width, img_height))

for frame in frames:
    video_writer.write(frame)

video_writer.release()
