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
from itertools import product
from skimage import transform as sktrans

img_size = 512

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

def get_landmarks_from_heatmap(heatmaps, project='default', name=None):
    ## Get landmarks from heatmaps with scaling ratios ##
    ## heatmaps: N x L x H x H ##
    ## ratio: N ##
    landmarks = []
    # Get x and y of landmarks
    for i in range(heatmaps.shape[0]):
        hm = heatmaps[i, :, :].cpu().numpy()
        # replace the channel of non-existent landmark with np.zeros
        # 不是很合理的方法，依据为"'不存在的landmark的预测heatmap最大值会很小'，正常max值>0.8，不存在的landmark的max值较小"
        # LAT侧位中仅有4张图像hm.max()大于0.60，最高0.61，且多是P4
        # AP正位中仅有43张hm.max()小于0.7，15张hm.max()小于0.6
        if  hm.max() < 0.7 and name is not None:
            print('%s: P%d, %f'% (name, i+1, hm.max()))
        if 'AP' in project and i==0 and hm.max() < 0.1:
            pos_y, pos_x = 0, 0
        # if 'LAT' in project and i==3:
        #     pos_yx = np.argmax(hm)
        #     pos_y, pos_x = np.unravel_index(pos_yx, hm.shape)
        #     landmarks.append([pos_y, pos_x])
        #     hm[pos_y, pos_x] = -999
        #     pos_yx = np.argmax(hm)
        #     pos_y, pos_x = np.unravel_index(pos_yx, hm.shape)
        #     landmarks.append([pos_y, pos_x])
        #     continue
        else:
            pos_yx = np.argmax(hm)
            pos_y, pos_x = np.unravel_index(pos_yx, hm.shape)
        landmarks.append([pos_y, pos_x])

    return landmarks

def visualize_heatmap(input, landmarks, landmarks_gt=None):
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
                         'P3-center of remote Ulna', 
                        #  'P5-center of proximal Ulna',
                         'P4-point of proximal part ']
        ldm_color_list = [(255, 0, 0), (51, 255, 255), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    # img = np.transpose(input.cpu().numpy())[:,:,0]
    if torch.is_tensor(input):
        img = input.cpu().numpy()[0, :, :]
        img = 255 * (img - np.min(img)) / (np.max(img) - np.min(img))
        img = img.astype(np.uint8)
        img = cv2.merge([img, img, img])
    else:
        img = input
    # draw landmarks on image
    if landmarks_gt is not None:
        for idx, (y_pos, x_pos) in enumerate(landmarks_gt):
            '''
                params:(image, (x_pos, y_pos), ...)
                original point: left upside, right -> X, down-> Y
            '''
            if y_pos == 0. and x_pos==0.:
                continue
            cv2.circle(img, (x_pos, y_pos), 3, (0, 255, 0), -1)

    lat_proximal = [0, 0]
    for idx, (y_pos, x_pos) in enumerate(landmarks):
        '''
            params:(image, (x_pos, y_pos), ...)
            original point: left upside, right -> X, down-> Y
        '''
        if y_pos == 0. and x_pos==0.:
            continue
        # get the midpoint in LAT
        if len(landmarks)==5 and (idx==2 or idx==4):
            lat_proximal[0] = int((lat_proximal[0] + y_pos) / (idx/2))
            lat_proximal[1] = int((lat_proximal[1] + x_pos) / (idx/2))
            if idx==2:
                continue
            if idx == 4:
                cv2.circle(img, (lat_proximal[1], lat_proximal[0]), 3, ldm_color_list[idx], -1)
                cv2.putText(img, ldm_name_list[idx], (lat_proximal[1]+10, lat_proximal[0]-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, ldm_color_list[idx], 1)  
                break
        cv2.circle(img, (x_pos, y_pos), 3, ldm_color_list[idx], -1)
        cv2.putText(img, ldm_name_list[idx], (x_pos+10, y_pos-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, ldm_color_list[idx], 1)    
    
    return img

def norm(x, vmin=None, vmax=None):
    if vmin is None or vmax is None:
        vmin, vmax = x.min(), x.max()
    else:
        x[x < vmin] = vmin
        x[x > vmax] = vmax
    if vmin == vmax:
        return x
    else:
        return (x-vmin)/(vmax-vmin)


def gaussianHeatmap(sigma, dim: int = 2, nsigma: int = 3):
    if nsigma <= 2:
        print('[Warning]: nsigma={} is recommended to be greater than 2'.format(nsigma))
    radius = round(nsigma*sigma)
    center = tuple([radius for i in range(dim)])
    mask_shape = tuple([2*radius for i in range(dim)])
    mask = np.zeros(mask_shape, dtype=float)
    sig2 = sigma**2
    coef = sigma*np.sqrt(2*np.pi)
    for p in product(*[range(i) for i in mask_shape]):
        d2 = sum((i-j)**2 for i, j in zip(center, p))
        mask[p] = np.exp(-d2/sig2/2)/coef
    mask = (mask-mask.min())/(mask.max()-mask.min()) # necessary?, yes, the output heatmap is processed with sigmoid

    def genHeatmap(point, shape):
        ret = np.zeros(shape, dtype=float)
        if point[0] <= 0 or point[1] <= 0 or point[0]>shape[0] or point[1]>shape[1]:
            return ret
        bboxs = [(max(0, point[ax]-radius), min(shape[ax], point[ax]+radius))
                 for ax in range(dim)]
        img_sls = tuple([slice(i, j) for i, j in bboxs])

        mask_begins = [max(0, radius-point[ax]) for ax in range(dim)]
        mask_sls = tuple([slice(beg, beg+j-i)
                          for beg, (i, j) in zip(mask_begins, bboxs)])

        ret[img_sls] = mask[mask_sls]
        return ret

    return genHeatmap

def merge_hm(map1, map2, shape):
    map = map1 + map2
    pos_exist = []
    for y in range(shape[0]):
        for x in range(shape[1]):
            if map1[y, x] != 0:
                pos_exist.append((y, x))

    for (y, x) in pos_exist:
        if map2[y, x] == 0:
            pos_exist.remove((y, x))

    for (y, x) in pos_exist:
        map[y, x] = max(map1[y, x], map2[y, x])

    return map

def rotate(img, angle):
    '''
        Rotate image by a certain angle around its center in counter-clockwise direction
    '''
    ret = sktrans.rotate(img, angle)
    return ret

def translate(img, offsets):
    ''' translation
        offsets: n-item list-like, for each dim
    '''
    offsets = tuple(offsets)
    # size = img.shape[1:]
    size = img.shape

    if offsets[0]<0:
        new_x_start = 0
        new_x_end = size[0] + offsets[0]
        old_x_start = -offsets[0]
        old_x_end   = size[0]
    else:
        new_x_start = offsets[0]
        new_x_end = size[0]
        old_x_start = 0
        old_x_end   = size[0] - offsets[0]

    if offsets[1]<0:
        new_y_start = 0
        new_y_end = size[1] + offsets[1]
        old_y_start = -offsets[1]
        old_y_end   = size[1]
    else:
        new_y_start = offsets[1]
        new_y_end = size[1]
        old_y_start = 0
        old_y_end   = size[1] - offsets[1]

    new_sls = tuple([slice(new_x_start, new_x_end), slice(new_y_start, new_y_end)])
    old_sls = tuple([slice(old_x_start, old_x_end), slice(old_y_start, old_y_end)])

    # ret = []
    # for old in img:
    #     new = np.zeros(size)
    #     new[new_sls] = old[old_sls]
    #     ret.append(new)
    # return np.array(ret)

    new = np.zeros(size)
    new[new_sls] = img[old_sls]
    return np.array(new)