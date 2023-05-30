'''
Date: 2023-05-24 14:19:57
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-05-30 10:24:17
FilePath: /QC-wrist/eval/char_recognize.py
Description: 
'''

import cv2
import os
import numpy as np

# 计算图片的相似度
def image_similarity(image1, image2):
    match = image1 == image2
    acc = match.sum() / match.size
    return acc

def is_position_mark(ori_img):
    is_existing = False
    # 模板字符
    template_R = 'tools/char-R.png'
    template_L = 'tools/char-L.png'

    # 图像二值化, 闭运算, 取连通域, 逐个对比
    _, th_img = cv2.threshold(ori_img, thresh=200, maxval=255, type=cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    close_img = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE, kernel)
    close_img = close_img.astype(np.uint8)

    # cv2.connectedComponentsWithStats requires gray image
    ret, _, stats, _ = cv2.connectedComponentsWithStats(close_img, connectivity=4)

    if ret < 1:
        return is_existing

    compared_res = []
    for idx, unit in enumerate(stats):
        x, y, w, h, _ = unit
        if x == y == 0:
            continue
        connected_component = th_img[y:y+h, x:x+w]
        connected_component = cv2.threshold(cv2.resize(connected_component, (64, 64)), 0, 255, cv2.THRESH_BINARY)[1]
        # cv2.imwrite(str(idx)+'.png', connected_component)
        # if position == 'R':
        acc1 = image_similarity(cv2.imread(filename=os.path.join(os.path.dirname(os.path.dirname(__file__)), template_R), flags=cv2.IMREAD_GRAYSCALE),
                                           connected_component)
        # elif position == 'L':
        acc2 = image_similarity(cv2.imread(filename=os.path.join(os.path.dirname(os.path.dirname(__file__)), template_L), flags=cv2.IMREAD_GRAYSCALE),
                                           connected_component)
        compared_res.append(max(acc1, acc2))

    for acc in compared_res:
        if acc > 0.90:
            is_existing = True
            break
    
    return is_existing

if __name__ == '__main__':
    
    path = '..//data/wrist_LAT'
    files = [i for i in os.listdir(path) if i.endswith('.png')]
    is_existing = []
    for i in files:
        is_existing.append(is_position_mark(cv2.imread(os.path.join(path, i), flags=cv2.IMREAD_GRAYSCALE)))
    cnt = 0
    for i, j in zip(files, is_existing):
        if j == False:
            # print(i)
            cnt += 1
    print(cnt)