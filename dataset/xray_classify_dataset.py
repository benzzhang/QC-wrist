'''
@Author     : Jian Zhang
@Init Date  : 2023-04-18 16:38
@File       : xray_classify_dataset.py
@IDE        : PyCharm
@Description: 0->[0, 1] 无异物伪影
              1->[1, 0] 有异物伪影
'''
import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

__all__ = ['XrayClassifyDataset']


class XrayClassifyDataset(Dataset):

    def __init__(self, img_list, meta, transform, prefix='data/', size=(512,512)):
        self.prefix = prefix
        # read imgs_list and metas
        imgs_list = open(img_list).readlines()
        self.imgs_list = [l.strip() for l in imgs_list]
        metas = open(meta).readlines()
        self.metas = [[int(i) for i in v.strip().split(' ')] for v in metas]
        self.transform = transform
        self.img_size = size

    def __getitem__(self, index):
        img_path = os.path.join(self.prefix, self.imgs_list[index].strip())
        img = cv2.imread(img_path)
        '''
            In order to adapt the in input of models, 
            : scale the size to the times of 32
            : scale the size to (960, 1920)
        '''
        # h = img.shape[0]
        # w = img.shape[1]
        # img_size = (w // 32 * 32, h // 32 * 32)
        img = cv2.resize(img, self.img_size)
        # cv2.imwrite('./'+self.imgs_list[index].strip(), img) 
        # print('resize:%d %d --> %d %d' % (h, w, img_size[1], img_size[0]))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform != None:
            img = self.transform(image=img)['image']
        img = torch.FloatTensor(img.transpose((2, 0, 1)))

        label = self.metas[index][1]
        if label == 0:
            label = [0, 1]
        else:
            label = [1, 0]
        label = torch.FloatTensor(label)

        return img, label

    def __len__(self):
        return len(self.imgs_list)
