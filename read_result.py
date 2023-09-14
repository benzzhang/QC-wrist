'''
Date: 2023-09-05 14:22:36
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-09-12 17:09:01
FilePath: /QC-wrist/read_result.py
Description: 
'''
import json
import os
dcms = [i for i in os.listdir('inference_task/') if i != '.gitkeep' ]
count = len(os.listdir('inference_task/'))
with open('inference_result/inference.json', 'r', encoding='utf-8') as json_file:
    json_dict = json.load(json_file)
    score = []
    for i in dcms:
        score.append(json_dict[i])
    A, B, C, D = [], [], [], []
    for i in score:
        if i < 60:
            D.append(i)
        elif i < 80:
            C.append(i)
        elif i < 90:
            B.append(i)
        else:
            A.append(i)
    print('A级：', len(A),'个， 分数：', A)
    print('B级：', len(B),'个， 分数：', B)
    print('C级：', len(C),'个， 分数：', C)
    print('D级：', len(D),'个， 分数：', D)
    print('合格率：', (len(A)+len(B)+len(C))/count)
    print('良好率：', (len(A)+len(B))/count)
    print('优秀率：', (len(A))/count)
