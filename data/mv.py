'''
Date: 2023-07-26 11:52:03
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-07-26 11:55:30
FilePath: /QC-wrist/data/mv.py
Description: 
'''
import os
import shutil

ap_files = [i for i in os.listdir('./wrist_LAT') if i.endswith('.png')]
for i in ap_files:
    shutil.copy(os.path.join('./wrist_LAT', i), os.path.join('./wrist', i))
    os.rename(os.path.join('./wrist', i), os.path.join('./wrist', i.replace('.png', '-LAT.png')))
