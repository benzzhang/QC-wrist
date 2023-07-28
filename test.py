'''
Date: 2023-07-27 16:09:43
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-07-27 17:18:17
FilePath: /QC-wrist/test.py
Description: 
'''
from utils import get_landmarks_from_heatmap, gaussianHeatmap, visualize_heatmap
import cv2
import numpy as np

img_size = [512, 512]

genHeatmap = gaussianHeatmap(5, dim=len((512, 512)))
lms_list = [[float(i) for i in v.strip().split(' ')] for v in open('data/wrist_AP_train_landmarks_list.txt').readlines()]
# lms_list = [[0, 0, 0, 0, 0, 0]]
for idx in range(0, 1):
    lms = lms_list[idx]
    lms = [[int(lms[i] * 1280), int(lms[i + 1] * 640)] for i in range(0, int(len(lms)), 2)]
    lms_mask = np.ones(len(lms))

    for i in range(len(lms)):
        if lms[i][0] > img_size[0] or lms[i][0] < 0:
            lms_mask[i] = 0
            continue
        if lms[i][1] > img_size[1] or lms[i][1] < 0:
            lms_mask[i] = 0
            continue

    lms_heatmap = [genHeatmap((x, y), (1280, 640)) for [x, y] in lms]
    for i in range(len(lms_heatmap)):
        landmarks = []
        # Get x and y of landmarks
        for j in range(len(lms_heatmap)):
            hm = lms_heatmap[j]
            pos_yx = np.argmax(hm)
            pos_y, pos_x = np.unravel_index(pos_yx, hm.shape)
            landmarks.append([pos_y, pos_x])
        visualize_img = visualize_heatmap(np.stack((lms_heatmap[i],) * 3, axis=-1), landmarks, None)
        cv2.imwrite('./2{}.png'.format(str(i)), visualize_img)