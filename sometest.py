'''
Date: 2023-05-19 10:16:34
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-05-24 10:04:37
FilePath: /QC-wrist/sometest.py
Description: 
'''
import cv2

with open('data/wrist_AP_class_list.txt', 'r') as f:
    metas = f.readlines()
    metass = [[int(i) for i in v.strip().split(' ')] for v in metas]
    for idx, i in enumerate(metass):
        if i[3] not in  [1, 2, 3]:
            print(idx)