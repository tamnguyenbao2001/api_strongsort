import cv2
import torch
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from collections import deque

def convert_to_xywh(x):
    # xywh = torch.zeros(output.shape[0], 4)
    # xywh[:,0] = output[:,0].cpu().float()*img_width
    # xywh[:,1] = output[:,1].cpu().float()*img_height
    # xywh[:,2] = output[:,2].cpu().float()*img_width - xywh[:,0]
    # xywh[:,3] = output[:,3].cpu().float()*img_height - xywh[:,1]
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    z = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:,0] = x[:,0].float()#*img_width
    y[:,1] = x[:,1].float()#*img_height
    y[:,2] = x[:,2].float()#*img_width 
    y[:,3] = x[:,3].float()#*img_height
    z[:,0] = ((y[:,0]+y[:,2])/2).round()
    z[:,1] = ((y[:,1]+y[:,3])/2).round()
    z[:,2] =(y[:,2] - y[:,0]).round()
    z[:,3] = (y[:,3] - y[:,1]).round()
    return z

def process_frame(frame, tracked_id_kpts, id_kpts, recognizer):   # tracked_id_kpts: list of [(id, corresponding_kpts)]
                                                           # id_kpts: dictionary of {id: {"kpts": [], "status": }}
    track_id_basket = []                                   # recognizer: transformer
    targets = []

    for track_id, kpts, coord in tracked_id_kpts:
        track_id_basket.append(track_id)
        if track_id in id_kpts:
            id_kpts[track_id]["kpts"].append(kpts)
            id_kpts[track_id]["coord"] = coord
            if len(id_kpts[track_id]["kpts"]) >= 5:
                targets.append((track_id, id_kpts[track_id]["kpts"]))
        else:
            id_kpts[track_id] = {"kpts": deque([kpts], maxlen=5), "status": "stacking",
            "coord": coord}

    if len(targets) > 0:
        flag = True
        idxes, values = tuple(zip(*targets))
        batch = tf.reshape(tf.convert_to_tensor(values), (len(idxes), 255))
        torch.set_printoptions(threshold=10000)
        try:
            preds = recognizer.predict(batch).argmax(1)
        except: flag = False
        if flag == True:
            for i in range(len(idxes)):
                if preds[i] == 0:
                    id_kpts[idxes[i]]["status"] = "Normal"
                elif preds[i] == 1:
                    id_kpts[idxes[i]]["status"] = "Fighting"
                else:
                    id_kpts[idxes[i]]["status"] = "Smoking"

    for id_kpt in list(id_kpts.keys()):  # Remove dead ids
        if id_kpt not in track_id_basket:
            del id_kpts[id_kpt]
            continue
        cv2.putText(frame, id_kpts[id_kpt]["status"], id_kpts[id_kpt]["coord"], 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return id_kpts

def patch_extractor(result):      # 10000 loops: 0.544s vs 1.0423s
    return tuple(map(lambda x: int(x[1]) if x[0]%2 == 0 else int(x[1]), enumerate(result.tolist())))[:4]

@tf.function(reduce_retracing = True)    # 100 loops 1.8s    vs   4.1718s
def distortion_free_resize(image, img_size):
    h, w = img_size
    image = tf.cast(tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True),
                    dtype=tf.uint8)

    # Check the amount of padding needed to be done
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Calculate padding values
    height = pad_height // 2
    pad_height_top = height + (pad_height % 2)
    pad_height_bottom = height
    width = pad_width // 2
    pad_width_left = width + (pad_width % 2)
    pad_width_right = width

    # Pad the image
    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )
    
    return image, pad_width_left, pad_width_right, pad_height_top, pad_height_bottom





EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def draw_connections(frame, keypoints, confidence_threshold, edges = EDGES):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)


def get_detector(model_dir):
    return torch.hub.load("ultralytics/yolov5", "custom", path = model_dir, force_reload=False)



