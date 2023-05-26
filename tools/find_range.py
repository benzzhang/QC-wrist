'''
Date: 2023-05-17 15:07:53
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-05-18 14:14:52
FilePath: /QC-wrist/utils/find_range.py
Description: 找出AP和LAT中图像尺寸除以320后的整数频次, 6和3最多
'''


import os
import cv2

path = '../data/wrist_AP'
img = [i for i in os.listdir(path) if '.png' in i]
dh = dict()
dw = dict()
for i in img:
    pic = cv2.imread(os.path.join(path, i))
    hh = pic.shape[0]
    ww = pic.shape[1]
    h = hh // 32 // 10
    w = ww // 32 // 10
    if str(h) in dh.keys():
        dh[str(h)] = dh[str(h)] + 1
    else:
        dh[str(h)] = 1
    
    if str(w) in dw.keys():
        dw[str(w)] = dw[str(w)] + 1
    else:
        dw[str(w)] = 1

print(dh)
print(dw)
