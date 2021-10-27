# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 19:37:48 2019

@author: 陈泽浩
"""

import os
import cv2
import numpy as np
import time
import argparse

import torch
from YOLO3.detector import YOLOv3
from sort import Sort
from util import COLORS_10, draw_bboxes, display_number, get_line, get_kb

class Detector(object):
    def __init__(self,display = False,write_video = False):
        
        self.yolo3 = YOLOv3(cfg = "YOLO3/cfg/yolov3-spp.cfg",data_cfg = "YOLO3/cfg/head.data", weights = 'YOLO3/best.pt', img_size = 960, is_xyxy=True)
        self.write_video = write_video
        self.display = display
        if self.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        self.vdo = cv2.VideoCapture()

    def open(self, video_path, write_path):
        assert os.path.isfile(video_path), "Error: path error"
        self.vdo.open(video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x1 = int(input('请输入计数线的起点x坐标,(0<=x<'+str(self.im_width)+'):'))
        y1 = int(input('请输入计数线的起点y坐标,(0<=y<'+str(self.im_height)+'):'))
        x2 = int(input('请输入计数线的终点x坐标,(0<=x<'+str(self.im_width)+'):'))
        y2 = int(input('请输入计数线的终点y坐标,(0<=y<'+str(self.im_height)+'):'))
        if x1 == x2:
            self.vertical = True
            self.k, self.b = 0, x1
        else:
            self.vertical = False
            self.k, self.b = get_kb((x1,y1),(x2,y2))
        self.counting_line_start_point, self.counting_line_end_point = get_line(self.k, self.b, self.im_width, self.im_height, self.vertical)
        self.sort = Sort(k = self.k, b = self.b)
        if self.write_video:
            fourcc =  cv2.VideoWriter_fourcc(*'mp4v')
            self.output = cv2.VideoWriter(write_path, fourcc, 25, (self.im_width,self.im_height))
        return self.vdo.isOpened()
        
    def detect(self):
#        xmin, ymin, xmax, ymax = self.area
        start_time = time.time()
        i = 0
        while self.vdo.grab():
#            start = time.time()
            _, ori_im = self.vdo.retrieve()
#            im = ori_im[ymin:ymax, xmin:xmax, (2,1,0)]
#            print(im.shape)
#            cv2.imshow('hello',ori_im)
            with torch.no_grad():
                bbox_xywh, cls_conf, cls_ids = self.yolo3(ori_im)
#            print(bbox_xywh)
            cls_conf = cls_conf.reshape(-1,1)
            det = np.concatenate((bbox_xywh,cls_conf),axis = 1)
            if bbox_xywh is not None:
#                mask = cls_ids==0
#                bbox_xywh = bbox_xywh[mask]
#                bbox_xywh[:,3] *= 1.2
#                cls_conf = cls_conf[mask]
                outputs = self.sort.update(det)
#                end = time.time()
#                print("time: {}s, fps: {}".format(end-start, 1/(end-start)))
                if (len(outputs) > 0) and (self.display or self.write_video):
                    bbox_xyxy = outputs[:,:4]
                    identities = outputs[:,-1]
                    ori_im = draw_bboxes(ori_im, bbox_xyxy, identities, offset=(0,0))
                    
            if (self.display or self.write_video):
                ori_im = display_number(ori_im, self.sort.forward_cnt, self.sort.backward_cnt, self.counting_line_start_point, self.counting_line_end_point)
            
            if self.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.write_video:
                self.output.write(ori_im)
        end_time = time.time()
        
        print("total time: {}s".format(end_time-start_time))
        print("正向人数: {}".format(self.sort.forward_cnt))
        print("反向人数: {}".format(self.sort.backward_cnt))
            
                
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='detection counting')
    parser.add_argument('--v', type=str, default='demo.mp4', help='video path to count')
    parser.add_argument('--d', type=bool, default=True, help='display counting video or not')
    parser.add_argument('--w', type=bool, default=True, help='save counting video or not')
    parser.add_argument('--o', type=str, default='output1.mp4', help='save video path')
    
    opt = parser.parse_args()

    det = Detector(display = opt.d,write_video = opt.w)
    det.open(video_path = opt.v, write_path = opt.o)
    det.detect()