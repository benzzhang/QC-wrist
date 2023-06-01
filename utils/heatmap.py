'''
@Author     : Jian Zhang
@Init Date  : 2023-04-20 09:49
@File       : heatmap.py
@IDE        : PyCharm
@Description: 
'''
# encoding: utf8
import torch
import numpy as np
from .misc import *
import cv2

img_size = 512

__all__ = ['make_image', 'visualize_heatmap','get_landmarks_from_heatmap', 'visualize_heatmap']


# functions to show an image
def make_image(img, mean=(0, 0, 0), std=(1, 1, 1)):
    for i in range(0, 3):
        img[i] = img[i] * std[i] + mean[i]  # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))


def gauss(x, a, b, c):
    return torch.exp(-torch.pow(torch.add(x, -b), 2).div(2 * c * c)).mul(a)


def colorize(x):
    ''' Converts a one-channel grayscale image to a color heatmap image '''
    if x.dim() == 2:
        torch.unsqueeze(x, 0, out=x)
    if x.dim() == 3:
        cl = torch.zeros([3, x.size(1), x.size(2)])
        cl[0] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
        cl[1] = gauss(x, 1, .5, .3)
        cl[2] = gauss(x, 1, .2, .3)
        cl[cl.gt(1)] = 1
    elif x.dim() == 4:
        cl = torch.zeros([x.size(0), 3, x.size(2), x.size(3)])
        cl[:, 0, :, :] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
        cl[:, 1, :, :] = gauss(x, 1, .5, .3)
        cl[:, 2, :, :] = gauss(x, 1, .2, .3)
    return cl

def get_landmarks_from_heatmap(heatmaps):
    ## Get landmarks from heatmaps with scaling ratios ##
    ## heatmaps: N x L x H x H ##
    ## ratio: N ##
    landmarks = []
    # Get x and y of landmarks
    for i in range(heatmaps.shape[0]):
        hm = heatmaps[i, :, :].cpu().numpy()
        pos_yx = np.argmax(hm)
        pos_y, pos_x = np.unravel_index(pos_yx, hm.shape)
        landmarks.append([pos_y, pos_x])

    return landmarks

def visualize_heatmap(input, landmarks, landmarks_gt):
    if len(landmarks) == 3:
        # ['P1-拇指指掌关节', 'P2-桡骨茎突', 'P3-尺骨茎突']
        ldm_name_list = ['P1-metacarpophalangeal joint', 
                         'P2-styloid process of Radius', 
                         'P3-styloid process of Ulna']
        ldm_color_list = [(255, 0, 0), (51, 255, 255), (0, 0, 255)]
    if len(landmarks) == 5:
        # ['P1-舟骨中心', 'P2-桡骨远端中心', 'P3-桡骨近端中心', 'P4-尺骨远端中心', 'P5-尺骨近端中心']
        ldm_name_list = ['P1-center of Scaphoid', 
                         'P2-center of remote Radius', 
                         'P3-center of proximal Radius', 
                         'P4-center of remote Ulna', 
                         'P5-center of proximal Ulna']
        ldm_color_list = [(255, 0, 0), (51, 255, 255), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    # img = np.transpose(input.cpu().numpy())[:,:,0]
    img = input.cpu().numpy()[0, :, :]
    img = 255 * (img - np.min(img)) / (np.max(img) - np.min(img))
    img = img.astype(np.uint8)
    img = cv2.merge([img, img, img])
    # draw landmarks on image
    for idx, (y_pos, x_pos) in enumerate(landmarks):
        '''
            params:(image, (x_pos, y_pos), ...)
            original point: left upside, right -> X, down-> Y
        '''
        cv2.circle(img, (x_pos, y_pos), 3, ldm_color_list[idx], -1)
        cv2.putText(img, ldm_name_list[idx], (x_pos+10, y_pos-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, ldm_color_list[idx], 1)    
    for idx, (y_pos, x_pos) in enumerate(landmarks_gt):
        '''
            params:(image, (x_pos, y_pos), ...)
            original point: left upside, right -> X, down-> Y
        '''
        cv2.circle(img, (x_pos, y_pos), 3, (0, 255, 0), -1)
    return img
