import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import glob
import torch
import tensorflow as tf
from collections import deque

from my_utils import patch_extractor, distortion_free_resize, draw_connections, process_frame
from my_utils import get_detector
from model import transformer
import onnxruntime



FONT = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = 1
COLOR = (255, 0, 0)
THICKNESS = 2
### yolov5 weight 
model_dir = "weights/best.onnx"

### Pose estimator

sess = onnxruntime.InferenceSession("movenet_thunder.onnx",providers= ['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape


model = torch.hub.load("ultralytics/yolov5", "custom", path = model_dir, force_reload= False)

### Action recognizer
recognizer = transformer()

# Video dir
test_video_dir = "IMG_2573.MOV"


# Visualise and output kpts of corresponding x1, y1, x2, y2
def visualisation(frame, x1, y1, x2, y2):                              # Input: x1, x2, y1, y2
    # Step 1: extract patch
    cropped_img, left, right, top, bot = distortion_free_resize(frame[y1:y2, x1:x2, [2, 1, 0]], (256, 256))
    # Step 2: Run pose estimator
    cropped_img = cropped_img.numpy()
    outputs_t = sess.run([], {input_name: cropped_img[np.newaxis, ...]})
    visualizing_keypoints_t = (outputs_t[0])
    # Step 3: Draw keypoints
    draw_connections(cropped_img, visualizing_keypoints_t, 0.25)
    # Step 4: Merging
    cropped_img = cropped_img[top:256-bot, left:256-right, :]
    cropped_img = cv2.resize(cropped_img, (x2 - x1, y2 - y1))
    frame[y1:y2, x1:x2, [2, 1, 0]] = cropped_img
    return visualizing_keypoints_t


cap = cv2.VideoCapture(test_video_dir)
colors_dict = {0: (0, 255, 255), 1: (0, 0, 255), 2: (255, 255, 255 ), 3: (255, 0, 0)}

id_kpts = {}

cnt = 0
ret, frame = cap.read()
while ret:
    start = time.time()
    id_deepsort = [1, 2, 3, 4,5,6,7,8,9]
    preds = model(frame)
    tracked_id_kpts = []
    for ith, pred in enumerate(preds.xyxyn[0]):
        x1, y1, x2, y2 = patch_extractor(pred)
            
        tracked_id_kpts.append((id_deepsort[ith], visualisation(frame, x1, y1, x2, y2)))        # (id_deepsort, keypoints)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors_dict[pred[-1].item()], 2)


    if cnt % 5 == 0:
        id_kpts = process_frame(tracked_id_kpts, id_kpts, recognizer)


    cv2.imshow("real-time", frame)
    
    if cv2.waitKey(10) == ord("q"):
        break
    ret, frame = cap.read()
    cnt +=1
    if cnt >= 11:
        cnt = 1
    print(time.time() - start)
    
        
cap.release()
cv2.destroyAllWindows()