'''
Date: 2023-05-26 14:08:13
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-05-29 11:17:37
FilePath: /QC-wrist/eval/point_judge.py
Description: TODO: 翻转图片到同一视角 --> 指掌关节位于左上角, 舟骨在上方

'''

import math
import numpy as np

def flip_AP(p1, p2, p3, size):
    if p2[1] < p1[1] and p3[1] < p1[1]:
        if (p2[0]+p3[0])/2 < p1[0]:
            p1[1] = size - p1[1]
            p2[1] = size - p2[1]
            p3[1] = size - p3[1]

        if (p2[0]+p3[0])/2 > p1[0]:
            p1 = (size[1]-p1[1], p1[0])
            p2 = (size[1]-p2[1], p2[0])
            p3 = (size[1]-p3[1], p3[0])
            size = (size[1], size[0])

    if p2[0] < p1[0] and p3[0] < p1[0]:
        p1 = (p1[1], size[0]-p1[0])
        p2 = (p2[1], size[0]-p2[0])
        p3 = (p3[1], size[0]-p3[0])
        size = (size[1], size[0])

    return p1, p2, p3, size

def flip_LAT(p1, p2, p3, p4, p5, size):
    if p1[1]>p2[1] and p2[1]>p3[1]:
        if (p2[1]-p3[1])>(size[1]*0.2):
            p1[1] = size - p1[1]
            p2[1] = size - p2[1]
            p3[1] = size - p3[1]
            p4[1] = size - p4[1]
            p5[1] = size - p5[1]

    if p1[0]<p2[0] and p2[0]<p3[0]:
        if (p3[0]-p2[0])>(size[0]*0.2):
            p1 = (size[1]-p1[1], p1[0])
            p2 = (size[1]-p2[1], p2[0])
            p3 = (size[1]-p3[1], p3[0])
            p4 = (size[1]-p4[1], p4[0])
            p5 = (size[1]-p5[1], p5[0])
            size = (size[1], size[0])

    if p1[0]>p2[0] and p2[0]>p3[0]:
        if (p2[0]-p1[0])>(size[0]*0.2):
            p1 = (p1[1], size[0]-p1[0])
            p2 = (p2[1], size[0]-p2[0])
            p3 = (p3[1], size[0]-p3[0])
            p4 = (p4[1], size[0]-p4[0])
            p5 = (p5[1], size[0]-p5[0])
            size = (size[1], size[0])

    return p1, p2, p3, p4, p5, size

'''
    尺桡骨茎突连线中点位于图像正中，上下、左右差<1cm
'''
def midpoint_of_StyloidProcess_is_center(p1, p2, pixelspacing, size):
    # midpoint: (x, y) = (w, h)
    midpoint = ( (p1[0]+p2[0])/2, (p1[1]+p2[1])/2 )
    # units: cm
    gap_x = abs(size(0) - 2*midpoint[0]) * pixelspacing[0] * 0.1
    gap_y = abs(size(1) - 2*midpoint[1]) * pixelspacing[1] * 0.1
    
    score = 0
    if gap_x < 1:
        score += 2.5
    if gap_y < 1:
        score += 2.5

    return score

'''
    尺桡骨茎突连线与图像纵轴垂直, 角度90°±5以内°
'''
def line_of_StyloidProcess_is_horizontal(p1, p2):

    vec = [abs(p1[0] - p2[0]),
           abs(p1[1] - p2[1])]

    angle_yaxis = np.rad2deg(math.atan(vec[0] / vec[1]))

    if angle_yaxis < 85:
        score = 0
    else:
        score = int(angle_yaxis - 85) + 1
    return score

'''
    下缘包含尺桡骨3-5cm
'''
def include_radius_ulna(p1, p2, pixelspacing, size):
    #TODO: 翻转图片到同一视角：指掌关节位于左上角
    # units: cm
    midpoint = ( (p1[0]+p2[0])/2, (p1[1]+p2[1])/2 )

    distance_from_lowest = (size[1] - midpoint[1]) * pixelspacing[1] * 0.1

    if distance_from_lowest < 3:
        score = 0
    if distance_from_lowest > 5:
        score = 0
    else:
        d1 = abs(distance_from_lowest - 3)
        d2 = abs(distance_from_lowest - 5)
        score = (1 - (abs(d1 - d2) / 2)) *5
        
    return score

'''
    左右最外侧距影像边缘3-5cm
'''
def distance_from_StyloidProcess_to_edge(p1, p2, pixelspacing, size):
    #TODO: 翻转图片到同一视角：指掌关节位于左上角
    # units: cm
    if p1[0] > p2[0]:
        l = p1
        r = p2
    else:
        l = p2
        r = p1
        
    distance_l = l[0] * pixelspacing[1] * 0.1
    distance_r = (size[0] - r[0]) * pixelspacing[1] * 0.1

    if distance_l < 3:
        score = 0
    if distance_l > 5:
        score = 0
    else:
        d1 = abs(distance_l - 3)
        d2 = abs(distance_l - 5)
        score = (1 - (abs(d1 - d2) / 2)) *2.5

    if distance_r < 3:
        score += 0
    if distance_r > 5:
        score += 0
    else:
        d1 = abs(distance_r - 3)
        d2 = abs(distance_r - 5)
        score += (1 - (abs(d1 - d2) / 2)) *2.5
        
    return score


'''
    舟骨位于图像正中, 上下、左右差<1cm
'''
def Scaphoid_is_center(p1, pixelspacing, size):
    # units: cm
    gap_x = abs(size(0) - 2*p1[0]) * pixelspacing[0] * 0.1
    gap_y = abs(size(1) - 2*p1[1]) * pixelspacing[1] * 0.1
    
    score = 0
    if gap_x < 1:
        score += 2.5
    if gap_y < 1:
        score += 2.5

    return score

'''
    腕关节长轴与影像长轴平行
'''
def line_of_LongAxis_is_vertical(p1, p2, p3):
    #TODO: 翻转图片到同一视角：指掌关节位于左上角
    # midpoint: (x, y) = (w, h)
    lowest_midpoint = ( (p2[0]+p3[0])/2, (p2[1]+p3[1])/2 )

    vec = [abs(p1[0] - lowest_midpoint[0]),
           abs(p1[1] - lowest_midpoint[1])]

    angle_yaxis = np.rad2deg(math.atan(vec[0] / vec[1]))

    if angle_yaxis > 10:
        score = 0
    else:
        score = int(10 - angle_yaxis) + 1
    return score