'''
@Author     : Jian Zhang
@Init Date  : 2023-04-20 10:36
@File       : xray_landmark_dataset.py
@IDE        : PyCharm
@Description: 
'''
import os
import numpy as np
import cv2
import math

import torch
from torch.utils.data import Dataset

from .util import gaussianHeatmap, rotate, translate

__all__ = ['WristLandmarkMaskDataset']


class WristLandmarkMaskDataset(Dataset):

    def __init__(self, img_list, meta, transform_paras, prefix='data/', size=(512, 512), sigma=5):

        self.transform_paras = transform_paras
        self.prefix = prefix
        self.img_size = size
        # read img_list and metas
        self.img_list = [l.strip() for l in open(img_list).readlines()]
        self.img_data_list = self.__readAllData__()
        # self.idx_of_lms = list(range(14)) + list(range(20, 26)) # 取部分points用
        self.idx_of_lms = list(range(3))
        self.lms_list = [[float(i) for i in v.strip().split(' ')] for v in open(meta).readlines()]
        self.genHeatmap = gaussianHeatmap(sigma, dim=len(size))


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
        origin_img = self.img_data_list[index]
        img_size = list(origin_img.shape)
        img = cv2.cvtColor(origin_img, cv2.COLOR_GRAY2RGB)
        '''
            .shape return: (height, width)
        '''
        h, w, _ = img.shape
        img = img.transpose((2, 0, 1))
        # get the heatmap of landmark in image
        lms = self.lms_list[index]
        lms = [(int(lms[i] * h), int(lms[i + 1] * w)) for i in range(0, int(len(lms)), 2)]
        # lms = [lms[idx] for idx in self.idx_of_lms] # 取部分points用
        lms_heatmap = [self.genHeatmap((x, y), (h, w)) for (x, y) in lms]
        lms_heatmap = np.array(lms_heatmap)

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
            if lms[i][0] > img_size[0] or lms[i][0] < 0:
                lms_mask[i] = 0
                continue
            if lms[i][1] > img_size[1] or lms[i][1] < 0:
                lms_mask[i] = 0
                continue
            if rotate_pos[i][0] > img_size[0] or rotate_pos[i][0] < 0:
                lms_mask[i] = 0
                continue
            if rotate_pos[i][1] > img_size[1] or rotate_pos[i][1] < 0:
                lms_mask[i] = 0
                continue
            if translate_pos[i][0] > img_size[0] or translate_pos[i][0] < 0:
                lms_mask[i] = 0
                continue
            if translate_pos[i][1] > img_size[1] or translate_pos[i][1] < 0:
                lms_mask[i] = 0
                continue

        return torch.FloatTensor(img), torch.FloatTensor(lms_heatmap), torch.FloatTensor(lms_mask), img_name

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    # prefix = '../data/wrist_AP'
    # img_list = '../data/wrist_AP_valid_list.txt'
    # meta = '../data/wrist_AP_valid_landmarks_list.txt'

    prefix = 'data/wrist_AP'
    img_list = 'data/wrist_AP_valid_list.txt'
    meta = 'data/wrist_AP_valid_landmarks_list.txt'

    transform_paras = {'rotate_angle': 30, 'offset': [50, 10]}
    wrist_dataset = WristLandmarkMaskDataset(img_list, meta, transform_paras, prefix, size=(960, 1920))

    for i in range(wrist_dataset.__len__()):
        image, lms_heatmap, lms_mask = wrist_dataset.__getitem__(i)
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
        cv2.imwrite('./heatmap'+ str(i)+'.png', lms_heatmap) 

        # for (x,y) in translate_pos:
        #     cv2.circle(image, (x,y), 4, (255,0,0), 2)
        # cv2.imshow('image', image)
        cv2.imwrite('./image'+ str(i)+'.png', image) 

        # key = cv2.waitKey(-1)
        # if key != 27:
        #     continue
        # else:
        #     exit(0)
