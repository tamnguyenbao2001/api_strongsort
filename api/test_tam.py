from re import T
import sys
import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
from pathlib import Path

import cv2
import torch
import numpy as np
from datetime import datetime

import pandas as pd
import warnings
warnings.filterwarnings('ignore')


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # strongsort root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str('C:/Users/ACER/Downloads/api/api/yolov8_tracking') not in sys.path:
    sys.path.append(str('C:/Users/ACER/Downloads/api/api/yolov8_tracking'))  # add yolov5 ROOT to PATH
if str('C:/Users/ACER/Downloads/api/api/yolov8_tracking/yolov8') not in sys.path:
    sys.path.append(str('C:/Users/ACER/Downloads/api/api/yolov8_tracking/yolov8')) 
if str('C:/Users/ACER/Downloads/api/api/yolov8_tracking/trackers/strongsort') not in sys.path:
    sys.path.append(str('C:/Users/ACER/Downloads/api/api/yolov8_tracking/trackers/strongsort'))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import threading
import boto3
import requests
import tensorflow as tf
from ActivityRecognition.my_utils import patch_extractor, distortion_free_resize, draw_connections, process_frame, convert_to_xywh
from ActivityRecognition.my_utils import get_detector
from ActivityRecognition.model import transformer
import onnxruntime
from flask import Flask, render_template, request, redirect, session
from trackers.multi_tracker_zoo import create_tracker
from ultralytics.yolo.utils.torch_utils import select_device
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = 1
COLOR = (255, 0, 0)
THICKNESS = 2
### yolov5 weight 
model_dir = "ActivityRecognition/weights/best.onnx"

model = torch.hub.load("ultralytics/yolov5", "custom", path = model_dir, force_reload= False)  

sess = onnxruntime.InferenceSession("ActivityRecognition/movenet_thunder.onnx",None, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
recognizer = transformer()

tracking_method = 'strongsort'
tracking_config = 'C:/Users/ACER/Downloads/api/api/yolov8_tracking/trackers/strongsort/configs/strongsort.yaml' 
reid_weights = Path('osnet_x0_25_msmt17.engine')
device = select_device('0')
print(device)
half = True
tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
if hasattr(tracker, 'model'):
    if hasattr(tracker.model, 'warmup'):
        tracker.model.warmup()
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

app = Flask(__name__)
@app.route('/strongsort',methods=['POST'])
def handleStrongSORT():
    input_txt = 'dataserver_1.txt'   #file txt: dataserver_1.txt
    video_txt = 'video_server_1.avi'  #file video video_server_1.avi
    a = open('eee.txt', 'w')
    id_kpts = {}
    names = {'worker': 1,'supervisor': 2,'CEO': 3,'Director': 4,'truck':5}
    cap = cv2.VideoCapture('data_server/Video/'+video_txt)
    img_height = int(cap.get(4))
    img_width = int(cap.get(3))
    cnt = 0
    outputs = [None]
    track_id = None
    curr_frames, prev_frames = [None], [None] 
    size = (img_width,img_height)
    list_data = []
    list_data_all = []
    num_frame = 0
    with open('data_server/Data/'+input_txt, 'r') as f:
        data = f.readlines()
    id_list = np.zeros(len(data),dtype = int)
    old_id = 0
    for dataline in data:
        datat = dataline.split(' ')[:-1]
        datat = [eval(i) for i in datat]
        if old_id == int(datat[0]):
            list_data.append(datat)
        else: 
            old_id +=1
            list_data = []
            list_data.append(datat)
            list_data_all.append(list_data)
    cap.release()
    cap = cv2.VideoCapture('data_server/Video/'+video_txt)
    while(cap.isOpened()):
        ret, im0 = cap.read()

        start_time = time.time()
        if ret == True:
            flag_out = True
            curr_frames = im0
            dataa = torch.Tensor(list_data_all[num_frame]).to(device)
            strongt = time.time()
            outputs = tracker.update(dataa.cpu(), im0)
            print(outputs)
            print("Strongsort time: ",time.time()-strongt)
            tracked_id_kpts = []         
            if len(outputs) == 0:
                continue
            try:
                for i in range(len(outputs)):
                    with open('qqq.txt', 'a') as f:
                        f.write(('%g ' * 9 + '\n') % (dataa[i,0],outputs[i,-3] ,outputs[i,-2],outputs[i,-1], outputs[i,0], outputs[i,1], outputs[i,2], outputs[i,3],names_action[id_kpts[i+1]["status"]]))
            except: pass
            cv2.imshow("TAM",im0)
            cv2.waitKey(10)
            num_frame +=1      
        else: break
    cap.release()
    return 'eee.txt'




@app.route('/action',methods=['POST'])
def handleAction():
    input_txt = 'dataserver_1.txt'   #file txt: dataserver_1.txt
    video_txt = 'video_server_1.avi'  #file video video_server_1.avi
    a = open('qqq.txt', 'w')
    id_kpts = {}
    names = {'worker': 1,'supervisor': 2,'CEO': 3,'Director': 4,'truck':5}
    names_action = {'stacking': 0,'Normal': 1,'Fighting': 2,'Smoking': 3}
    cap = cv2.VideoCapture('data_server/Video/'+video_txt)
    img_height = int(cap.get(4))
    img_width = int(cap.get(3))
    cnt = 0
    ret, im0 = cap.read()
    preds = model(im0[:, :, [2, 1, 0]])
    id_keypoints = {ith: [] for ith in range(len(preds.xyxyn[0]))}
    temp = []
    outputs = [None]
    track_id = None
    curr_frames, prev_frames = [None], [None] 
    size = (img_width,img_height)
    list_data = []
    list_data_all = []
    num_frame = 0
    with open('data_server/Data/'+input_txt, 'r') as f:
        data = f.readlines()
    id_list = np.zeros(len(data),dtype = int)
    old_id = 0
    for dataline in data:
        datat = dataline.split(' ')[:-1]
        datat = [eval(i) for i in datat]
        if old_id == int(datat[0]):
            list_data.append(datat)
        else: 
            old_id +=1
            list_data = []
            list_data.append(datat)
            list_data_all.append(list_data)
    cap.release()
    cap = cv2.VideoCapture('data_server/Video/'+video_txt)
    while(cap.isOpened()):
        ret, im0 = cap.read()

        start_time = time.time()
        if ret == True:
            flag_out = True
            curr_frames = im0
            dataa = torch.Tensor(list_data_all[num_frame]).to(device)
            strongt = time.time()
            outputs = tracker.update(dataa.cpu(), im0)
            print(outputs)
            print("Strongsort time: ",time.time()-strongt)
            tracked_id_kpts = []         
            if len(outputs) == 0:
                flag_out = False
            if flag_out:
                id_deepsort = outputs[:,4]
                outputs = torch.Tensor(outputs)
                for ith, pred in enumerate(outputs):
                    x1, y1, x2, y2 = patch_extractor(pred)
                    try:
                        tracked_id_kpts.append((id_deepsort[ith], visualisation(im0, x1, y1, x2, y2), (x1, y1-20)))        # (id_deepsort, keypoints)
                    except: continue
                    # cv2.rectangle(im0, (x1, y1), (x2, y2), colors_dict[pred[-2].item()], 2)
                if cnt % 5 == 0:
                    id_kpts = process_frame(im0,tracked_id_kpts, id_kpts, recognizer)
                if cnt >= 11:
                    cnt = 1
                try:
                    for i in range(len(outputs)):
                        with open('qqq.txt', 'a') as f:
                            f.write(('%g ' * 9 + '\n') % (dataa[i,0],outputs[i,-3] ,outputs[i,-2],outputs[i,-1], outputs[i,0], outputs[i,1], outputs[i,2], outputs[i,3],names_action[id_kpts[i+1]["status"]]))
                except: pass
            cnt +=1
            cv2.imshow("TAM",im0)
            cv2.waitKey(10)
            num_frame +=1      
        else: break
    cap.release()
    return 'qqq.txt'

if __name__ == '__main__':
    app.run()