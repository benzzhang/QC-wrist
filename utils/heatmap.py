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
import math

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

def visualize_in_evaluate(input, landmarks, pixelspacing):
    # OpenCV B,G,R
    if len(landmarks) == 3:
        # ['P1-拇指指掌关节', 'P2-桡骨茎突', 'P3-尺骨茎突']
        ldm_name_list = ['P1', 
                         'P2', 
                         'P3']
        ldm_color_list = [(0, 0, 255), (0, 0, 255), (0, 0, 255)]
    if len(landmarks) == 5:
        # ['P1-舟骨中心', 'P2-桡骨远端中心', 'P3-桡骨近端中心', 'P4-尺骨远端中心', 'P5-尺骨近端中心']
        # 模型实际会预测5个点，但是展示只用4个点，只显示[0,1,3]和 midpoint of [2,4]
        ldm_name_list = ['P1', 
                         'P2', 
                         '',
                         'P3', 
                         'P4' # combine with proximal radius to cal the midpoint
                         ]
        ldm_color_list = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255)]
    # img = np.transpose(input.cpu().numpy())[:,:,0]
    if torch.is_tensor(input):
        img = input.cpu().numpy()[0, :, :]
        img = 255 * (img - np.min(img)) / (np.max(img) - np.min(img))
        img = img.astype(np.uint8)
        img = cv2.merge([img, img, img])
    else:
        img = input
    
    lat_proximal = [0, 0]
    # draw the key points first, then write out the measurement information
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
                if idx == 2:
                    continue
                if idx == 4:
                    cv2.circle(img, (lat_proximal[1], lat_proximal[0]), 5, ldm_color_list[idx], -1)
                    cv2.putText(img, ldm_name_list[idx], (lat_proximal[1]+10, lat_proximal[0]-10), cv2.FONT_HERSHEY_COMPLEX, 1, ldm_color_list[idx], 1)  
                    continue
            cv2.circle(img, (x_pos, y_pos), 5, ldm_color_list[idx], -1)
            cv2.putText(img, ldm_name_list[idx], (x_pos+10, y_pos-10), cv2.FONT_HERSHEY_COMPLEX, 1, ldm_color_list[idx], 1)    

    '''
        landmarks的坐标:(y, x), y轴从上到下, x轴从左到右
        cv2作图时坐标系: (x, y),x轴从左向右, y轴从上到下
        两个坐标原点都是左上角
        所以在landmarks不作更改的情况下, 于cv2作图时, 需要将landmarks坐标对调
    '''

    def calculate_angle(point_1, point_2, point_3):
        """
        根据三点坐标计算夹角
        :param point_1: 点1坐标
        :param point_2: 点2坐标
        :param point_3: 点3坐标
        :return: 返回指定角的夹角值
        """
        a=math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
        b=math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
        c=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1]))
        B=math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
        return B
    
    def DoubleArrowedLine(img, start, end, eval_color, pixelspacing):
        # one line
        cv2.line(img, (start[1], start[0]), (end[1], end[0]), eval_color, thickness=2)
        # two branches
        if start[1] == end[1]:
            if end[0] == 0:
                # top: 在cv2坐标系中, end已经落位, 对end的y都要增加, x一增一减
                cv2.line(img, (end[1], end[0]), (end[1]-10, end[0]+10), (0, 0, 255), thickness=2)
                cv2.line(img, (end[1], end[0]), (end[1]+10, end[0]+10), (0, 0, 255), thickness=2)
                cv2.putText(img, "{:.2f}mm".format((start[0]-end[0])*pixelspacing), ( int((start[1]+end[1])/2), int((start[0]+end[0])/2) ), cv2.FONT_HERSHEY_COMPLEX, 1, eval_color, 1)
            else:
                # down
                cv2.line(img, (end[1], end[0]), (end[1]-10, end[0]-10), (0, 0, 255), thickness=2)
                cv2.line(img, (end[1], end[0]), (end[1]+10, end[0]-10), (0, 0, 255), thickness=2)
                cv2.putText(img, "{:.2f}mm".format((end[0]-start[0])*pixelspacing), ( int((start[1]+end[1])/2), int((start[0]+end[0])/2) ), cv2.FONT_HERSHEY_COMPLEX, 1, eval_color, 1)
        if start[0] == end[0]:
            if end[1] == 0:
                # left
                cv2.line(img, (end[1], end[0]), (end[1]+10, end[0]-10), (0, 0, 255), thickness=2)
                cv2.line(img, (end[1], end[0]), (end[1]+10, end[0]+10), (0, 0, 255), thickness=2)
                cv2.putText(img, "{:.2f}mm".format((start[1]-end[1])*pixelspacing), ( int((start[1]+end[1])/2), int((start[0]+end[0])/2)-10 ), cv2.FONT_HERSHEY_COMPLEX, 1, eval_color, 1)
            else:
                # right
                cv2.line(img, (end[1], end[0]), (end[1]-10, end[0]-10), (0, 0, 255), thickness=2)
                cv2.line(img, (end[1], end[0]), (end[1]-10, end[0]+10), (0, 0, 255), thickness=2)
                cv2.putText(img, "{:.2f}mm".format((end[1]-start[1])*pixelspacing), ( int((start[1]+end[1])/2), int((start[0]+end[0])/2)-10 ), cv2.FONT_HERSHEY_COMPLEX, 1, eval_color, 1)

    eval_color = (0, 255, 0)
    y_pixelspacing = pixelspacing[0]
    x_pixelspacing = pixelspacing[1]

    if len(landmarks) == 3:
        p1 = landmarks[0]
        p2 = landmarks[1]
        p3 = landmarks[2]
        ap_midpoint = ( int((p2[0]+p3[0])/2), int((p2[1]+p3[1])/2) )

        # line of [p2 to p3]
        cv2.line(img, (p2[1], p2[0]), (p3[1], p3[0]), (0, 0, 255), thickness=1)
        DoubleArrowedLine(img, ap_midpoint, (0, ap_midpoint[1]), (0, 0, 255), pixelspacing=y_pixelspacing) # from midpoint to top
        DoubleArrowedLine(img, ap_midpoint, (img.shape[0], ap_midpoint[1]), (0, 0, 255), pixelspacing=y_pixelspacing) # from midpoint to bottom
        DoubleArrowedLine(img, ap_midpoint, (ap_midpoint[0], 0), (0, 0, 255), pixelspacing=x_pixelspacing) # from midpoint to left
        DoubleArrowedLine(img, ap_midpoint, (ap_midpoint[0], img.shape[1]), (0, 0, 255), pixelspacing=x_pixelspacing) # from midpoint to right
        # X-cross from midpoint
        for point in[(ap_midpoint[1]+10, ap_midpoint[0]+10), 
                     (ap_midpoint[1]-10, ap_midpoint[0]+10), 
                     (ap_midpoint[1]-10, ap_midpoint[0]-10), 
                     (ap_midpoint[1]+10, ap_midpoint[0]-10)]:
            cv2.line(img, (ap_midpoint[1], ap_midpoint[0]), point, (0, 0, 255), thickness=2) # bottom right
            cv2.line(img, (ap_midpoint[1], ap_midpoint[0]), point, (0, 0, 255), thickness=2) # bottom left
            cv2.line(img, (ap_midpoint[1], ap_midpoint[0]), point, (0, 0, 255), thickness=2) # top left
            cv2.line(img, (ap_midpoint[1], ap_midpoint[0]), point, (0, 0, 255), thickness=2) # top right

        # this means that it's vertical
        if img.shape[0] >= img.shape[1]:
            if p2[1] < p3[1]:
                l, r = p2, p3
            else:
                l, r = p3, p2
            DoubleArrowedLine(img, l, (l[0], 0), eval_color, pixelspacing=x_pixelspacing) # left to edge
            cv2.line(img, (l[1], l[0]), (l[1]-10, l[0]+10), (0, 0, 255), thickness=2) # left back to 'l'
            cv2.line(img, (l[1], l[0]), (l[1]-10, l[0]-10), (0, 0, 255), thickness=2) # left back to 'l'
            DoubleArrowedLine(img, r, (r[0], img.shape[1]), eval_color, pixelspacing=x_pixelspacing) # right to edge
            cv2.line(img, (r[1], r[0]), (r[1]+10, r[0]+10), (0, 0, 255), thickness=2) # right back to 'r'
            cv2.line(img, (r[1], r[0]), (r[1]+10, r[0]-10), (0, 0, 255), thickness=2) # right back to 'r'

            if r[0] > ap_midpoint[0]: # The right point is below the midpoint in cv2-coordinate-system
                start_angle = calculate_angle(r, ap_midpoint, (ap_midpoint[0], img.shape[1])) # cal angle: r-midpoint-right
                end_angle = 90
            else:
                start_angle = -calculate_angle(r, ap_midpoint, (ap_midpoint[0], img.shape[1])) # cal angle: r-midpoint-right
                end_angle = -90
            cv2.ellipse(img, (ap_midpoint[1], ap_midpoint[0]), (30, 30), 0, start_angle, end_angle, (0, 255, 0), 1)
            cv2.putText(img, "{:d} deg".format(int(abs(end_angle-start_angle))), ( ap_midpoint[1]-30, ap_midpoint[0]-30 ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, eval_color, 1) # write angle
        
        # this means that it's horizontal
        if img.shape[0] < img.shape[1]:
            if p2[0] < p3[0]:
                t, b = p2, p3
            else:
                t, b = p3, p2
            DoubleArrowedLine(img, t, (0, t[1]), eval_color, pixelspacing=y_pixelspacing) # up to top
            cv2.line(img, (t[1], t[0]), (t[1]+10, t[0]-10), (0, 0, 255), thickness=2) # top back to 't'
            cv2.line(img, (t[1], t[0]), (t[1]-10, t[0]-10), (0, 0, 255), thickness=2) # top back to 't'
            DoubleArrowedLine(img, b, (img.shape[0], b[1]), eval_color, pixelspacing=y_pixelspacing) # down to bottom
            cv2.line(img, (b[1], b[0]), (b[1]-10, b[0]+10), (0, 0, 255), thickness=2) # down back to 'b'
            cv2.line(img, (b[1], b[0]), (b[1]+10, b[0]+10), (0, 0, 255), thickness=2) # down back to 'b'

            #TODO:只从top点开始画，根据和垂直线的关系决定往左还是往右画弧
        cv2.circle(img, (ap_midpoint[1], ap_midpoint[0]), 5, [255, 255, 255], -1)


    if len(landmarks)==5:
        p1 = landmarks[0]
        p2 = landmarks[1]
        p3 = landmarks[2]
        p4 = landmarks[3]
        p5 = landmarks[4]
        lat_proximal = ( int((p3[0]+p5[0])/2), int((p3[1]+p5[1])/2) )
        
        DoubleArrowedLine(img, p1, (0, p1[1]), eval_color, pixelspacing=y_pixelspacing) # from p1 to top
        DoubleArrowedLine(img, p1, (img.shape[0], p1[1]), eval_color, pixelspacing=y_pixelspacing) # from p1 to down
        DoubleArrowedLine(img, p1, (p1[0], 0), eval_color, pixelspacing=x_pixelspacing) # from p1 to left
        DoubleArrowedLine(img, p1, (p1[0], img.shape[1]), eval_color, pixelspacing=x_pixelspacing) # from p1 to right

        # X-cross from p1
        for point in[(p1[1]+10, p1[0]+10), 
                     (p1[1]-10, p1[0]+10), 
                     (p1[1]-10, p1[0]-10), 
                     (p1[1]+10, p1[0]-10)]:
            cv2.line(img, (p1[1], p1[0]), point, (0, 0, 255), thickness=2) # bottom right
            cv2.line(img, (p1[1], p1[0]), point, (0, 0, 255), thickness=2) # bottom left
            cv2.line(img, (p1[1], p1[0]), point, (0, 0, 255), thickness=2) # top left
            cv2.line(img, (p1[1], p1[0]), point, (0, 0, 255), thickness=2) # top right

        cv2.line(img, (p1[1], p1[0]), (lat_proximal[1], lat_proximal[0]), (0, 0, 255), thickness=2) # from p1 to lat_proximal
        cv2.line(img, (p1[1], p1[0]), (p2[1], p2[0]), (0, 0, 255), thickness=2) # from p1 to p2
        cv2.line(img, (p1[1], p1[0]), (p4[1], p4[0]), (0, 0, 255), thickness=2) # from p1 to p4

        # write the angle truthfully without requiring acute angles
        if lat_proximal[0] > p1[0]: # lat_proximal is below p1
            start_angle = calculate_angle(lat_proximal, p1, (p1[0], img.shape[1])) # cal angle: lat_proximal-p1-right
            end_angle = 90
            cv2.ellipse(img, (p1[1], p1[0]), (50, 50), 0, start_angle, end_angle, (0, 255, 0), 1)
            cv2.putText(img, "{:d} deg".format(int(abs(end_angle-start_angle))), ( p1[1]+50, p1[0]+50 ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, eval_color, 1) # write angle
        if lat_proximal[0] < p1[0]: # lat_proximal is above p1
            start_angle = -calculate_angle(lat_proximal, p1, (p1[0], img.shape[1])) # cal angle: lat_proximal-p1-right
            end_angle = -90
            cv2.ellipse(img, (p1[1], p1[0]), (50, 50), 0, start_angle, end_angle, (0, 255, 0), 1)
            cv2.putText(img, "{:d} deg".format(int(abs(end_angle-start_angle))), ( p1[1]-50, p1[0]-50 ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, eval_color, 1) # write angle 
    
    return img

def visualize_heatmap(input, landmarks, landmarks_gt=None):
    if len(landmarks) == 3:
        # ['P1-拇指指掌关节', 'P2-桡骨茎突', 'P3-尺骨茎突']
        ldm_name_list = ['P1-metacarpophalangeal joint', 
                         'P2-styloid process of Radius', 
                         'P3-styloid process of Ulna']
        ldm_color_list = [(0, 0, 255), (255, 0, 0),  (255, 255, 0)]
    if len(landmarks) == 5:
        # ['P1-舟骨中心', 'P2-桡骨远端中心', 'P3-桡骨近端中心', 'P4-尺骨远端中心', 'P5-尺骨近端中心']
        # 模型实际会预测5个点，但是展示只用4个点，只显示[0,1,3]和 midpoint of [2,4]        
        ldm_name_list = ['P1-center of Scaphoid', 
                         'P2-center of remote Radius', 
                         'center of proximal Radius',
                         'P3-center of remote Ulna',
                         'P4-point of proximal part' # combine with proximal radius to cal the midpoint
                         ]
        ldm_color_list = [(0, 0, 255), (255, 0, 0), (0, 0, 0), (255, 255, 0), (51, 255, 255)]
    # img = np.transpose(input.cpu().numpy())[:,:,0]
    if torch.is_tensor(input):
        img = input.cpu().numpy()[0, :, :]
        img = 255 * (img - np.min(img)) / (np.max(img) - np.min(img))
        img = img.astype(np.uint8)
        img = cv2.merge([img, img, img])
    else:
        img = input
    # draw landmarks on image
    lat_proximal = [0, 0]
    if landmarks_gt is not None:
        for idx, (y_pos, x_pos) in enumerate(landmarks_gt):
            '''
                params:(image, (x_pos, y_pos), ...)
                original point: left upside, right -> X, down-> Y
            '''
            if y_pos == 0. and x_pos==0.:
                continue
            # get the midpoint in LAT
            if len(landmarks_gt)==5 and (idx==2 or idx==4):
                lat_proximal[0] = int((lat_proximal[0] + y_pos) / (idx/2))
                lat_proximal[1] = int((lat_proximal[1] + x_pos) / (idx/2))
                if idx == 2:
                    continue
                if idx == 4:
                    cv2.circle(img, (lat_proximal[1], lat_proximal[0]), 3, (0, 255, 0), -1)
                    break
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
            if idx == 2:
                continue
            if idx == 4:
                cv2.circle(img, (lat_proximal[1], lat_proximal[0]), 3, ldm_color_list[idx], -1)
                cv2.putText(img, ldm_name_list[idx], (lat_proximal[1]+10, lat_proximal[0]-10), cv2.FONT_HERSHEY_COMPLEX, 1, ldm_color_list[idx], 1)  
                break
        cv2.circle(img, (x_pos, y_pos), 3, ldm_color_list[idx], -1)
        cv2.putText(img, ldm_name_list[idx], (x_pos+10, y_pos-10), cv2.FONT_HERSHEY_COMPLEX, 1, ldm_color_list[idx], 1)    
    
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