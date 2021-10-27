# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:15:54 2021

@author: Misaka Violet

你指尖跃动的电光，是我此生不变的信仰，唯我超电磁炮永世长存
君指先跃动の光は，私の一生不变の信仰に，唯私の超电磁炮永世生き
The electric sparkle glittering on your fingertips is my rock solid faith for life, only my Railgun lives forever
"""

import cv2
import os
import numpy as np

class VideoWriter:
    def __init__(self, name, width, height, fps=25):
        # type: (str, int, int, int) -> None
        if not name.endswith('.mp4'):  # 保证文件名的后缀是.mp4
            name += '.mp4'
            warnings.warn('video name should ends with ".mp4"')
        self.__name = name          # 文件名
        self.__height = height      # 高
        self.__width = width        # 宽
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 如果是mp4视频，编码需要为mp4v
        self.__writer = cv2.VideoWriter(name, fourcc, fps, (width, height))

    def write(self, frame):
        if frame.dtype != np.uint8:  # 检查frame的类型
            raise ValueError('frame.dtype should be np.uint8')
        # 检查frame的大小
        row, col, _ = frame.shape
        if row != self.__height or col != self.__width:
            warnings.warn('长和宽不等于创建视频写入时的设置，此frame不会被写入视频')
            return
        self.__writer.write(frame)

    def close(self):
        self.__writer.release()

cap = cv2.VideoCapture('E:/yolov3_sort_count/demo.mpg')
cnt = 0

width = 1920
height = 1080
vw = VideoWriter('E:/yolov3_sort_count/demo.mp4', width, height)

while cap.isOpened():
    ret, frame = cap.read()
    
    if frame is None:
        break
    vw.write(frame)
    cnt +=1
  
vw.close()
cap.release()