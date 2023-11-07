'''
Date: 2023-05-26 14:08:13
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-11-07 11:25:09
FilePath: /QC-wrist/eval/point_judge.py
Description:

'''

import math
import numpy as np

'''
    上缘是否包含拇指指掌关节
'''
def metacarpophalangeal_joint_is_included(p1):
    if p1[0]==0 and p1[1]==0:
        return 0
    else:
        return 5

'''
    尺桡骨茎突连线中点是否位于图像正中(上下、左右差值<1cm), 
    超出时按超出比例减少获得的分数(超出比例最大100%)
'''
def midpoint_of_StyloidProcess_is_center(p1, p2, pixelspacing, size):
    # midpoint: (x, y) = (w, h)
    midpoint = ( (p1[0]+p2[0])/2, (p1[1]+p2[1])/2 )
    # units: cm
    gap0 = abs(size[0] - 2*midpoint[0]) * pixelspacing[0] * 0.1
    gap1 = abs(size[1] - 2*midpoint[1]) * pixelspacing[1] * 0.1
    
    score = 0
    if gap0 <= 1:
        score += 5
    if 1 < gap0 < 2:
        score += int(4 * (2-gap0)) + 1

    if gap1 <= 1:
        score += 5
    if 1 < gap1 < 2:
        score += int(4 * (2-gap1)) + 1

    return score, round(gap0, 3), round(gap1, 3)

'''
    尺桡骨茎突连线是否与影像纵轴垂直(角度90°±5°以内)
'''
def line_of_StyloidProcess_is_horizontal(p1, p2, size):

    vec = [abs(p1[0] - p2[0]),
           abs(p1[1] - p2[1])]

    if size[0] >= size[1]:
        try:
            angle_yaxis = np.rad2deg(math.atan(vec[0] / vec[1]))
        except:
            angle_yaxis = np.NaN
    else:
        try:
            angle_yaxis = np.rad2deg(math.atan(vec[1] / vec[0]))
        except:
            angle_yaxis = np.NaN

    score = 0
    if not np.isnan(angle_yaxis):
        if int(angle_yaxis) <= 5:
            score = 10
        if 5 < angle_yaxis < 10:
            score = int( 9 * (10-angle_yaxis)/5 ) + 1
        return score, int(angle_yaxis)
    else:
        return 0, 'NaN'
    
'''
    下缘是否包含尺桡骨3-5cm
'''
def include_radius_ulna(p1, p2, p3, pixelspacing, size):
    # units: cm
    midpoint = ( (p2[0]+p3[0])/2, (p2[1]+p3[1])/2 )

    if size[0] >= size[1]:
        # 判断竖直正方向
        if p1[0] > midpoint[0]: # 尺桡骨从上往下
            distance_from_lowest = midpoint[0] * pixelspacing[0] * 0.1
        else:
            distance_from_lowest = (size[0] - midpoint[0]) * pixelspacing[0] * 0.1
    else:
        # 判断水平正方向
        if p1[1] > midpoint[0]: # 尺桡骨从左往右
            distance_from_lowest = midpoint[1] * pixelspacing[1] * 0.1
        else:
            distance_from_lowest = (size[1] - midpoint[1]) * pixelspacing[1] * 0.1

    score = 0
    if 3 <= distance_from_lowest <= 5:
        score = 5
    if 2 < distance_from_lowest < 3:
        score = int(4 * (distance_from_lowest-2)) + 1
    if 5 < distance_from_lowest < 6:
        score = int(4 * (6-distance_from_lowest)) + 1

    return score, round(distance_from_lowest, 3)

'''
    左右最外侧是否距影像边缘3-5cm
'''
def distance_from_StyloidProcess_to_edge(p1, p2, pixelspacing, size):
    # units: cm

    if size[0] >= size[1]:
        if p1[1] > p2[1]:
            r = p1
            l = p2
        else:
            r = p2
            l = p1
            
        distance0 = l[1] * pixelspacing[1] * 0.1
        distance1 = (size[1] - r[1]) * pixelspacing[1] * 0.1

        score = 0
        if 3 <= distance0 <= 5:
            score += 3
        if 2 < distance0 < 3:
            score += int(2 * (distance0-2)) + 1
        if 5 < distance0 < 6:
            score += int(2 * (6-distance0)) + 1

        if 3 <= distance1 <= 5:
            score += 3
        if 2 < distance1 < 3:
            score += int(2 * (distance1-2)) + 1
        if 5 < distance1 < 6:
            score += int(2 * (6-distance1)) + 1

    else:
        if p1[1] > p2[1]:
            b = p1
            t = p2
        else:
            b = p2
            t = p1
            
        distance0 = t[0] * pixelspacing[0] * 0.1
        distance1 = (size[0] - b[0]) * pixelspacing[0] * 0.1

        score = 0
        if 3 <= distance0 <= 5:
            score += 3
        if 2 < distance0 < 3:
            score += int(2 * (distance0-2)) + 1
        if 5 < distance0 < 6:
            score += int(2 * (6-distance0)) + 1

        if 3 <= distance1 <= 5:
            score += 3
        if 2 < distance1 < 3:
            score += int(2 * (distance1-2)) + 1
        if 5 < distance1 < 6:
            score += int(2 * (6-distance1)) + 1
            
    return score, round(distance0, 3), round(distance1, 3)


'''
    舟骨是否位于图像正中(上下、左右差值<1cm)
'''
def Scaphoid_is_center(p1, pixelspacing, size):
    # units: cm
    gap0 = abs(size[0] - 2*p1[0]) * pixelspacing[0] * 0.1
    gap1 = abs(size[1] - 2*p1[1]) * pixelspacing[1] * 0.1
    
    score = 0
    if gap0 <= 1:
        score += 6
    if 1 < gap0 < 2:
        score += int(5 * (2-gap0)) + 1

    if gap1 <= 1:
        score += 6
    if 1 < gap1 < 2:
        score += int(5 * (2-gap1)) + 1

    return score, round(gap0, 3), round(gap1, 3)

'''
    腕关节长轴是否与影像纵轴平行(两轴所夹锐角<10°)
    · 如果两条线段之间的夹角在 0 到 5 度之间，可以认为它们非常接近平行。
    · 如果夹角在 5 到 10 度之间，可以认为它们相对平行，但可能有轻微的偏离。
    · 如果夹角在 10 到 15 度之间，可以认为它们大致平行，但具有一定程度的偏离。
    · 如果夹角在 15 到 20 度之间，可以认为它们基本平行，但存在明显的偏离。
'''
def line_of_LongAxis_is_vertical(p1, p2, p3, size):
    # midpoint: (x, y) = (w, h)
    lat_proximal = ( (p2[0]+p3[0])/2, (p2[1]+p3[1])/2 )

    vec = [abs(p1[0] - lat_proximal[0]),
           abs(p1[1] - lat_proximal[1])]

    if size[0] >= size[1]:
        try:
            angle_yaxis = np.rad2deg(math.atan(vec[1] / vec[0]))
        except:
            angle_yaxis = np.NaN
    else:
        try:
            angle_yaxis = np.rad2deg(math.atan(vec[0] / vec[1]))
        except:
            angle_yaxis = np.NaN
            
    score = 0
    if not np.isnan(angle_yaxis):
        if 0 <= int(angle_yaxis) <= 5:
            score = 12
        if 5 < int(angle_yaxis) <10:
            score = int(11 * (10 - angle_yaxis) / 5) + 1
        return score, int(angle_yaxis)
    else:
        return 0, 'NaN'
    
'''
    尺桡骨重叠(abandoned)
'''
def radius_and_ulna_overlap(p1, p2, p3, p4):

    # l1 [xa, ya, xb, yb]   l2 [xa, ya, xb, yb]
    # reference: https://blog.csdn.net/weixin_42990464/article/details/120652439
    def Intersect(l1, l2):
        v1 = (l1[0] - l2[0], l1[1] - l2[1])
        v2 = (l1[0] - l2[2], l1[1] - l2[3])
        v0 = (l1[0] - l1[2], l1[1] - l1[3])
        a = v0[0] * v1[1] - v0[1] * v1[0]
        b = v0[0] * v2[1] - v0[1] * v2[0]

        temp = l1
        l1 = l2
        l2 = temp
        v1 = (l1[0] - l2[0], l1[1] - l2[1])
        v2 = (l1[0] - l2[2], l1[1] - l2[3])
        v0 = (l1[0] - l1[2], l1[1] - l1[3])
        c = v0[0] * v1[1] - v0[1] * v1[0]
        d = v0[0] * v2[1] - v0[1] * v2[0]

        if a*b < 0 and c*d < 0:
            return True
        else:
            return False

    l1 = [p1[0], p1[1], p2[0], p2[1]]
    l2 = [p3[0], p3[1], p4[0], p4[1]]

    is_intersecting = Intersect(l1, l2)

    if is_intersecting:
        score = 9
    else:
        score = 0
    return score

'''
    尺桡骨远端重叠
'''
def distal_radius_and_ulna_overlap(p1, p2, p3):
    
    a=math.sqrt((p2[0]-p3[0])*(p2[0]-p3[0])+(p2[1]-p3[1])*(p2[1] - p3[1]))
    b=math.sqrt((p1[0]-p3[0])*(p1[0]-p3[0])+(p1[1]-p3[1])*(p1[1] - p3[1]))
    c=math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))
    try:
        angleA=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))
    except:
        angleA=np.NaN

    score = 0
    if not np.isnan(angleA):
        if angleA <= 45:
            score = 12
        if 45 < angleA < 60:
            score = int(11 * (60-angleA) / 15) + 1
        return score, math.ceil(angleA)
    else:
        return 0, 'NaN'