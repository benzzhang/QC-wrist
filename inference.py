'''
Date: 2023-05-26 10:19:09
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-05-29 18:10:17
FilePath: /QC-wrist/inference.py
Description: 
'''

import os
import yaml
import os
import shutil
import time
import yaml
import numpy as np
import pydicom

import torch
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data.dataloader import default_collate

import models
import dataset
from utils import Logger, AverageMeter, mkdir_p, progress_bar, visualize_heatmap, get_landmarks_from_heatmap
from eval import is_position_mark, flip_AP, flip_LAT, midpoint_of_StyloidProcess_is_center, line_of_LongAxis_is_vertical,\
include_radius_ulna, distance_from_StyloidProcess_to_edge, Scaphoid_is_center, line_of_StyloidProcess_is_horizontal, basic_information_completed, dose
import losses
import cv2

def init_ai_quality_model():
    '''
        initial the ai quality model
    '''
    global use_cuda
    model_classify = models.__dict__[config['arch_classify']](num_classes=config['num_classes_classify'])
    model_landmark_AP = models.__dict__[config['arch_lanmarks']](num_classes=config['num_classes_AP'],
                                                     local_net=config['local_net'])
    model_landmark_LAT = models.__dict__[config['arch_lanmarks']](num_classes=config['num_classes_LAT'],
                                                     local_net=config['local_net'])
    model_classify.load_state_dict(torch.load(os.path.join(config['save_path_classify'], 'model_best.pth.tar'))['state_dict'], strict=False)
    model_landmark_AP.load_state_dict(torch.load(os.path.join(config['save_path_AP'], 'model_best.pth.tar'))['state_dict'], strict=False)
    model_landmark_LAT.load_state_dict(torch.load(os.path.join(config['save_path_LAT'], 'model_best.pth.tar'))['state_dict'], strict=False)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model_classify = model_classify.cuda()
        model_landmark_AP = model_landmark_AP.cuda()
        model_landmark_LAT = model_landmark_LAT.cuda()
    return model_classify, model_landmark_AP, model_landmark_LAT


def inference(models, prending_list):
    '''
        evaluate the dicom by check tags completation and ai_quality_model, return the detail scores
    '''
    models[0].eval()
    models[1].eval()
    models[2].eval()
    print('start inferencing ...')
    res_list = []
    for i in prending_list:
        dcmfile = os.path.join(config['dcmfile_path'], i)
        df = pydicom.read_file(dcmfile, force=True)

        df.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        df_pixel = df.pixel_array
        scaled_df_pixel = (df_pixel - min(df_pixel.flatten())) / (max(df_pixel.flatten()) - min(df_pixel.flatten()))
        scaled_df_pixel *= 255
        resized_df0 = cv2.resize(scaled_df_pixel, (config['size_classify']['W'], config['size_classify']['H']))
        resized_df1 = cv2.resize(scaled_df_pixel, (config['size_landmarks']['W'], config['size_landmarks']['H']))

        resized_df0 = np.stack((resized_df0,) * 3, axis=-1)
        resized_df1 = np.stack((resized_df1,) * 3, axis=-1)

        df_tensor0 = torch.FloatTensor(np.expand_dims(resized_df0.transpose((2, 0, 1)), 0))
        df_tensor1 = torch.FloatTensor(np.expand_dims(resized_df1.transpose((2, 0, 1)), 0))

        if use_cuda:
            df_tensor0 = df_tensor0.cuda()
            df_tensor1 = df_tensor1.cuda()
            df_tensor0 = torch.autograd.Variable(df_tensor0)
            df_tensor1 = torch.autograd.Variable(df_tensor1)

        ProtocolName = df.data_element('ProtocolName').value

        # position mark detection
        res_mark = is_position_mark(scaled_df_pixel)
        # classify
        res_clsaaify = models[0](df_tensor0)
        # landmark
        if ProtocolName == '腕关节正位':
            res_landmark = models[1](df_tensor1)
        elif ProtocolName == '腕关节侧位':
            res_landmark = models[2](df_tensor1)

        '''
            return: True/False, True/False, List
        '''
        result = (res_mark, res_clsaaify[0][0].item() > res_clsaaify[0][1].item(), get_landmarks_from_heatmap(res_landmark.squeeze().detach()))
        res_list.append(result)

    return res_list

def evaluate_each(dcmfile, coordinate):
    df = pydicom.read_file(dcmfile, force=True)
    ProtocolName = df.data_element('ProtocolName').value
    PixelSpacing = df.data_element('PixelSpacing').value
    size = df.pixel_array.size

    layout_score = None
    if ProtocolName == '腕关节正位' or len(coordinate) == 3:
        p1 = coordinate[0]
        p2 = coordinate[1]
        p3 = coordinate[2]

        p1, p2, p3 = flip_AP(p1, p2, p3, size)

        layout_score = 0
        layout_score += midpoint_of_StyloidProcess_is_center(p2, p3, PixelSpacing, size)
        layout_score += line_of_StyloidProcess_is_horizontal(p2, p3)
        layout_score += include_radius_ulna(p2, p3, PixelSpacing, size)
        layout_score += distance_from_StyloidProcess_to_edge(p2, p3, PixelSpacing, size)

    if ProtocolName == '腕关节侧位' or len(coordinate) == 5:
        p1 = coordinate[0]
        p2 = coordinate[1]
        p3 = coordinate[2]
        p4 = coordinate[3]
        p5 = coordinate[4]

        p1, p2, p3, p4, p5 = flip_LAT(p1, p2, p3, p4, p5, size)

        layout_score = 0
        layout_score += Scaphoid_is_center(p1, PixelSpacing, size)
        layout_score += line_of_LongAxis_is_vertical(p1, p3, p5)

    layout_score += basic_information_completed(df)
    layout_score += dose(df)
    return layout_score


def main():
    import argparse
    global config
    
    parser = argparse.ArgumentParser(description='workflow of QC in wrist')
    # model related, including  Architecture, path, datasets
    parser.add_argument('--config-file', type=str, default='experiments/config_inference.yaml')
    parser.add_argument('--gpu-id', type=str, default='1,2')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    models = init_ai_quality_model()

    '''
        获取待推理影像列表
    '''
    prending_list = os.listdir(config['dcmfile_path'])
    res_inference = inference(models, prending_list)

    res_dict = dict()
    for res, path in zip(res_inference, prending_list):
        qualified_flag = True
        dcmfile = os.path.join(config['dcmfile_path'], path)
        if not res[0]:
            qualified_flag = False
        if res[1]:
            layout_score = 5
        else:
            layout_score = 10
        layout_score += evaluate_each(dcmfile, res[2])

        if qualified_flag:
            res_dict[str(path)] = layout_score
        else:
            res_dict[str(path)] = 0

    import openpyxl
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(list(res_dict))
    workbook.save('inference.xls')
    
    print('------------------- inference completed at {}-------------------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))

if __name__ == '__main__':
    main()