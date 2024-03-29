'''
@Author     : Jian Zhang
@Init Date  : 2023-04-20 10:36
@File       : xray_landmark_dataset.py
@IDE        : PyCharm
@Description: 
'''
import os, inspect
import numpy as np
import cv2
import math
import sys
import torch
from torch.utils.data import Dataset
from utils import gaussianHeatmap, rotate, translate, get_landmarks_from_heatmap, merge_hm
import albumentations as A

__all__ = ['WristLandmarkMaskDataset']


class WristLandmarkMaskDataset(Dataset):

    def __init__(self, img_list, meta, transform_paras, prefix='data/', size=(512, 512), sigma=5, merge=False):

        self.transform_paras = transform_paras
        self.prefix = prefix
        self.img_size = size
        # read img_list and metas
        self.img_list = [l.strip() for l in open(img_list).readlines()]
        self.img_data_list = self.__readAllData__()
        # self.idx_of_lms = list(range(14)) + list(range(20, 26)) # Select a section of points to use
        self.idx_of_lms = list(range(3))
        self.lms_list = [[float(i) for i in v.strip().split(' ')] for v in open(meta).readlines()]
        self.genHeatmap = gaussianHeatmap(sigma, dim=len(self.img_size))
        self.merge = merge

    def __readAllData__(self):
        img_data_list = []
        for index in range(len(self.img_list)):
            img_path = os.path.join(self.prefix, self.img_list[index].strip())
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            '''
                In order to adapt the in input of models, 
                : scale the size to the times of 32
                : scale the size to (960, 1920)
            '''
            # h = img.shape[0]
            # w = img.shape[1]
            # img_size = (w // 32 * 32, h // 32 * 32)
            img = cv2.resize(img, self.img_size)
            # cv2.imwrite('./'+self.img_list[index].strip(), img) 
            img_data_list.append(img)
        return img_data_list

    def __getitem__(self, index):
        img_name = self.img_list[index].strip()
        img = self.img_data_list[index]
        img_size = list(img.shape)
        '''
            .shape return: (height, width)
        '''
        h, w = img.shape
        # get the heatmap of landmark in image
        lms = self.lms_list[index]
        lms = [[int(lms[i] * h), int(lms[i + 1] * w)] for i in range(0, int(len(lms)), 2)]
        # lms = [lms[idx] for idx in self.idx_of_lms] # 取部分points用
        lms_heatmap = [self.genHeatmap((x, y), (h, w)) for [x, y] in lms]
        # the order of points in merge_heatmap is [p1, p2, p4, [p3, p5]] in LAT
        if self.merge:
            lms_heatmap = [lms_heatmap[0], lms_heatmap[1], lms_heatmap[3], merge_hm(lms_heatmap[2], lms_heatmap[4], (h, w))]
        # cv2 (H,W,C) ; tensor (N,C,H,W)
        lms_heatmap = np.array(lms_heatmap).transpose((1, 2, 0))

        if self.transform_paras['rotate_angle'] != 0 and self.transform_paras['offset'] != [0, 0]:
            # VerticalFlip
            if np.random.rand() < 0.5:
                VerticalFlip0 = A.VerticalFlip(always_apply=False, p=1)(image=img)
                VerticalFlip1 = A.VerticalFlip(always_apply=False, p=1)(image=lms_heatmap)
                img = VerticalFlip0['image']
                lms_heatmap = VerticalFlip1['image']
                lms = [[h-lms[i][0], lms[i][1]] for i in range(len(lms))]

            # HorizontalFlip
            if np.random.rand() < 0.5:
                HorizontalFlip0 = A.HorizontalFlip(always_apply=False, p=1)(image=img)
                HorizontalFlip1 = A.HorizontalFlip(always_apply=False, p=1)(image=lms_heatmap)
                img = HorizontalFlip0['image']
                lms_heatmap = HorizontalFlip1['image']
                lms = [[lms[i][0], w-lms[i][1]] for i in range(len(lms))]

        # get rotate positions
        def lms_rotate(points_list, center, angle):
            """
                Rotate points counterclockwise by a given angle around the center.
                The angle should be given in radians.
            """
            ox, oy = center
            angle = -np.deg2rad(angle)
            x_arr = np.array([p[0] for p in points_list])
            y_arr = np.array([p[1] for p in points_list])
            rotate_x_arr = ox + math.cos(angle) * (x_arr - ox) - math.sin(angle) * (y_arr - oy)
            rotate_y_arr = oy + math.sin(angle) * (x_arr - ox) + math.cos(angle) * (y_arr - oy)
            rotate_pos = [(int(x), int(y)) for x, y in zip(rotate_x_arr, rotate_y_arr)]
            return rotate_pos

        # transform use rotate and translate
        angle = (np.random.rand() - 0.5) * self.transform_paras['rotate_angle']
        offset_x = int((np.random.rand() - 0.5) * self.transform_paras['offset'][0])
        offset_y = int((np.random.rand() - 0.5) * self.transform_paras['offset'][1])
        
        # rotate
        if np.random.rand() < 0.5:
            img = rotate(img, angle)
            lms_heatmap = rotate(lms_heatmap, angle)
            rotate_pos = lms_rotate(lms, (img_size[0] / 2, img_size[1] / 2), angle)
        else:
            rotate_pos = lms
        # mask those
        if np.random.rand() < 0.5:
            img = translate(img, [offset_x, offset_y])
            lms_heatmap = translate(lms_heatmap, [offset_x, offset_y])
            translate_pos = [(_x + offset_x, _y + offset_y) for (_x, _y) in rotate_pos]
        else:
            translate_pos = rotate_pos

        # scale to [-1, 1]
        if np.max(img) > 1:
            img = (img - 127.5) / 127.5
        else:
            img = (img - 0.5) * 2

        lms_mask = np.ones(len(translate_pos))

        # mask those points that exceed the boundary of the image
        for i in range(len(translate_pos)):
            if lms[i][0] >= img_size[0] or lms[i][0] <= 0 or \
                lms[i][1] >= img_size[1] or lms[i][1] <= 0 or \
                rotate_pos[i][0] >= img_size[0] or rotate_pos[i][0] <= 0 or \
                rotate_pos[i][1] >= img_size[1] or rotate_pos[i][1] <= 0 or \
                translate_pos[i][0] >= img_size[0] or translate_pos[i][0] <= 0 or \
                translate_pos[i][1] >= img_size[1] or translate_pos[i][1] <= 0:
                lms_mask[i] = 0
        if self.merge:
            merge_mask = lms_mask[2] or lms_mask[4]
            lms_mask = np.delete(lms_mask, 2)
            lms_mask = np.delete(lms_mask, 3)
            lms_mask = np.append(lms_mask, merge_mask)

        img = cv2.merge([img, img, img])
        img = img.transpose((2, 0, 1))
        lms_heatmap = lms_heatmap.transpose((2, 0, 1))
        return torch.FloatTensor(img), torch.FloatTensor(lms_heatmap), torch.FloatTensor(lms_mask), img_name

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":

    prefix = '../data/wrist_LAT'
    img_list = '../data/wrist_LAT_valid_list.txt'
    meta = '../data/wrist_LAT_valid_landmarks_list.txt'

    transform_paras = {'rotate_angle': 30, 'offset': [30, 30]}
    wrist_dataset = WristLandmarkMaskDataset(img_list, meta, transform_paras, prefix, size=(640, 1280))

    for i in range(wrist_dataset.__len__()):
        image, lms_heatmap, lms_mask, img_name = wrist_dataset.__getitem__(i)
        image, lms_heatmap = image.numpy(), lms_heatmap.numpy()
        print(f'max: {np.max(image)}, min:{np.min(image)}, mean:{np.mean(image)}')

        image = np.transpose(image, (1, 2, 0)) * 255
        image = image.astype(np.uint8)
        image = cv2.merge([image[:, :, 0], image[:, :, 0], image[:, :, 0]])

        lms_heatmap = np.sum(lms_heatmap, axis=0) * 255
        lms_heatmap[lms_heatmap > 255] = 255
        lms_heatmap = lms_heatmap.astype(np.uint8)

        # for (x,y) in transform_pos:
        #     cv2.circle(lms_heatmap, (x,y), 1, (255,0,0), 1)
        # cv2.imshow('heatmap', lms_heatmap)
        # cv2.imwrite('./heatmap'+ str(i)+'.png', lms_heatmap) 

        # for (x,y) in translate_pos:
        #     cv2.circle(image, (x,y), 4, (255,0,0), 2)
        # cv2.imshow('image', image)
        # cv2.imwrite('./image'+ str(i)+'.png', image) 

        # key = cv2.waitKey(-1)
        # if key != 27:
        #     continue
        # else:
        #     exit(0)
