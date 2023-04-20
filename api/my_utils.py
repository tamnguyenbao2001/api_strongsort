import cv2
import torch
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub


def patch_extractor(result, img_width, img_height):
    res1 = int(result[0].item())
    res2 = int(result[1].item())
    res3 = int(result[2].item())
    res4 = int(result[3].item())
    return res1, res2, res3, res4

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

def distortion_free_resize(image, img_size):
    h, w = img_size
    image = tf.cast(tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True),
                    dtype=tf.int32)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

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

def get_pose_estimator():
    model_thunder = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    movenet_thunder = model_thunder.signatures['serving_default']
    return movenet_thunder
