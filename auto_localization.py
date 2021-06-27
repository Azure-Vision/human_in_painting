import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import baiduAPI
import IPython


def auto_local(background=None, human=None, use_labelmap=True, bottom_local=True):
    # parameter
    #background_path = './pics/background.png'
    #human_path = './pics/human.png'
    #use_labelmap = False

    #background = cv2.imread(background_path, 0)
    #human = cv2.imread(human_path)
    background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    if use_labelmap:
        labelmap = baiduAPI.getLabelMap("Images/tmp/cleaned_human.jpg")
    else:
        labelmap = np.ones(human.shape, np.uint8) * 255
    labelmap = labelmap.sum(axis=-1)
    labelmap = np.where(labelmap > 0, 1, 0).astype(np.uint8)

    sobelx = cv2.Sobel(background, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(background, cv2.CV_64F, 0, 1, ksize=5)
    energy = np.sqrt(np.square(sobelx) + np.square(sobely)).astype(int)
    energy = np.clip(energy, 0, np.mean(energy) * 3)

    # 图片右下角的坐标
    y_max = energy.shape[0] + human.shape[0]
    x_max = energy.shape[1] + human.shape[1]

    # grid search
    y_interval = 10
    x_interval = 10
    scale_interval = 20

    best_y, best_x, best_scale = 0, 0, 1.0
    lowest_energy = np.max(energy)

    for y_idx in range(y_interval):
        for x_idx in range(x_interval):
            for scale_idx in range(int(scale_interval / 3),  scale_interval):
                size = (1.0 / scale_interval) * (scale_idx + 1)
                _labelmap = cv2.resize(
                    labelmap.copy(), (0, 0), fx=size, fy=size)
                # 左上角
                y_start = int((background.shape[0] / y_interval) * y_idx)
                x_start = int((background.shape[1] / x_interval) * x_idx)
                if bottom_local:
                    y_start = background.shape[0] - _labelmap.shape[0]
                    if y_start < 0:
                        y_start = 0
                    if x_start < 0:
                        x_start = 0
                    if x_start > background.shape[1] - _labelmap.shape[1]:
                        continue

                y_range = min(
                    energy.shape[0] - y_start, _labelmap.shape[0])
                x_range = min(
                    energy.shape[1] - x_start, _labelmap.shape[1])
                # print(x_range, y_range, x_start, y_start,
                #   energy.shape, _labelmap.shape)
                _energy = np.mean(energy[y_start: y_start + y_range, x_start: x_start + x_range]
                                  * _labelmap[0: y_range, 0: x_range])
                if _energy < lowest_energy:
                    lowest_energy = _energy
                    best_scale = size
                    best_y = y_start + _labelmap.shape[0]
                    best_x = x_start + _labelmap.shape[1]

    return best_y, best_x, best_scale
