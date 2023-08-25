'''
Date: 2023-07-26 11:52:03
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-08-09 16:48:20
FilePath: /QC-wrist/data/cp_mv.py
Description: 
'''
import os
import shutil

position = 'AP'
format = '.json'
files = [i for i in os.listdir('./wrist_%s' %(position)) if i.endswith(format)]
for i in files:
    shutil.copy(os.path.join('./wrist_%s' %(position), i), os.path.join('./wrist', i))
    os.rename(os.path.join('./wrist', i), os.path.join('./wrist', i.replace(format, '-%s%s' %(position, format))))
