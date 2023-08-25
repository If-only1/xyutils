__all__ = ['VideoWriter']

import logging

import cv2
import numpy as np


class VideoWriter:
    def __init__(self, name, width, height, fps=25):
        # type: (str, int, int, int) -> None
        if not name.endswith('.mp4'):  # 保证文件名的后缀是.mp4
            name += '.mp4'
            logging.warning('video name should ends with ".mp4"')
        self.__name = name  # 文件名
        self.__height = height  # 高
        self.__width = width  # 宽
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        # fourcc = cv2.VideoWriter_fourcc(*'H264')
        self.__writer = cv2.VideoWriter(name, fourcc, fps, (width, height))

    def write(self, frame):
        if frame.dtype != np.uint8:  # 检查frame的类型
            raise ValueError('frame.dtype should be np.uint8')
        # 检查frame的大小
        row, col, _ = frame.shape
        if row != self.__height or col != self.__width:
            logging.warning('shape not be save,this frame will not be writen!')
            return
        self.__writer.write(frame)

    def close(self):
        self.__writer.release()
