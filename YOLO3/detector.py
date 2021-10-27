# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 18:22:00 2019

@author: 陈泽浩
"""

import torch
import numpy as np
import cv2

from sys import platform
from YOLO3.models import *
from YOLO3.utils.datasets import *
from YOLO3.utils.utils import *
from YOLO3.utils.torch_utils import *
import time

class YOLOv3(object):
    def __init__(self, cfg, data_cfg, weights, img_size = 960, is_xyxy=True):
        self.device = torch_utils.select_device()
        torch.backends.cudnn.benchmark = False  # set False for reproducible results
        
        # model definition
        self.model = Darknet(cfg, img_size)
        
        # Load weights
        self.model.load_state_dict(torch.load(weights, map_location=self.device)['model'])


        # Fuse Conv2d + BatchNorm2d layers
        self.model.fuse()
        
        # Eval mode
        self.model.to(self.device).eval()

        # constants
        self.size = img_size
        self.conf_thres = 0.6
        self.nms_thres = 0.2
        self.is_xyxy = is_xyxy

    def __call__(self, ori_img):
        # img to tensor
        img, *_ = letterbox(ori_img, new_shape=self.size)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0
        
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)

        pred, _ = self.model(img)
        
        det = non_max_suppression(pred, self.conf_thres, self.nms_thres)[0]
   
        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], ori_img.shape).round()
                    
#        print(det)            
        height , width = ori_img.shape[:2]
        det = det.cpu()
        det = det.data.numpy()
        bbox = np.empty_like(det[:,:4])
        if self.is_xyxy:
            # bbox x y x y
            bbox[:,0] = det[:,0]
            bbox[:,1] = det[:,1]
            bbox[:,2] = det[:,2]
            bbox[:,3] = det[:,3]
        else:
            # bbox x y w h
            bbox[:,0] = (det[:,0] + det[:,2]) / 2.0
            bbox[:,1] = (det[:,1] + det[:,3]) / 2.0
            bbox[:,2] = (det[:,2]-det[:,0])
            bbox[:,3] = (det[:,3]-det[:,1])
        cls_conf = det[:,5]
        cls_ids = det[:,6]
        return bbox, cls_conf, cls_ids

def letterbox(img, new_shape=416, color=(127.5, 127.5, 127.5), mode='auto'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    ratiow, ratioh = ratio, ratio
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    # Compute padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':  # square
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
    elif mode is 'scaleFill':
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape, new_shape)
        ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratiow, ratioh, dw, dh
  
def _main():
    
    cfg = 'cfg/yolov3-spp.cfg'
    data_cfg = "cfg/head.data"
    weights = "best.pt"
    img_size = 960
    path = "D:/data/picture/"
    
    yolo3 = YOLOv3(cfg, data_cfg, weights, img_size, is_xyxy=False)
    import os
    files = [os.path.join(path,file) for file in os.listdir(path)]
    files.sort()
    for filename in files:
        img = cv2.imread(filename)
        
        yolo3(img)
