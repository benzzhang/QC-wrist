'''
Date: 2023-05-26 10:19:09
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-05-26 16:54:14
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
from eval import char_recognize
import losses
import cv2

def init_ai_quality_model():
    '''
        initial the ai quality model
    '''
    global use_cuda
    model_classify = models.__dict__[config['arch']](num_classes=config['num_classes'])
    model_landmark_AP = models.__dict__[config['arch']](num_classes=config['num_classes'],
                                                     local_net=config['local_net'])
    model_landmark_LAT = models.__dict__[config['arch']](num_classes=config['num_classes'],
                                                     local_net=config['local_net'])
    model_classify.load_state_dict(torch.load(config['pretrained_weights'])['state_dict'], strict=True)
    model_landmark_AP.load_state_dict(torch.load(config['pretrained_weights'])['state_dict'], strict=True)
    model_landmark_LAT.load_state_dict(torch.load(config['pretrained_weights'])['state_dict'], strict=True)

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
    print('start inferencing ...')
    res_list = []
    for i in prending_list:
        dcmfile = os.path.join(config['dcmfile_path'], i)
        df = pydicom.read_file(dcmfile, force=True)

        df.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        df_pixel = df.pixel_array
        scaled_df_pixel = (df_pixel - min(df_pixel.flatten())) / (max(df_pixel.flatten()) - min(df_pixel.flatten()))
        scaled_df_pixel *= 255
        df_tensor = torch.FloatTensor(scaled_df_pixel.transpose((2, 0, 1)))

        ProtocolName = df.data_element('ProtocolName').value

        # position mark detection
        res_mark = char_recognize(scaled_df_pixel)
        # classify
        models[0].eval()
        res_clsaaify = models[0](df_tensor)
        # landmark
        if ProtocolName == '腕关节正位':
            res_landmark = models[1](df_tensor)
        elif ProtocolName == '腕关节侧位':
            res_landmark = models[2](df_tensor)

        result = (res_mark, res_clsaaify[0] > res_clsaaify[1], get_landmarks_from_heatmap(res_landmark.detach()))
        res_list.append(result)

    return res_list

def evaluate_each(dcmfile, coordinate):
    df = pydicom.read_file(dcmfile, force=True)
    ProtocolName = df.data_element('ProtocolName').value
    PixelSpacing = df.data_element('PixelSpacing').value
    if ProtocolName == '腕关节正位' or len(coordinate) == 3:
        p1 = coordinate[0]
        p2 = coordinate[1]
        p3 = coordinate[2]
    if ProtocolName == '腕关节侧位' or len(coordinate) == 5:
        p1 = coordinate[0]
        p2 = coordinate[1]
        p3 = coordinate[2]
        p4 = coordinate[3]
        p5 = coordinate[4]


def main():
    import argparse
    global config
    
    parser = argparse.ArgumentParser(description='workflow of QC in wrist')
    # model related, including  Architecture, path, datasets
    parser.add_argument('--config-file', type=str, default='experiments/config_inference.yaml')
    parser.add_argument('--gpu-id', type=str, default='2')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main(args.config_file)
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    models = init_ai_quality_model(config)

    '''
        获取待推理影像列表
    '''
    prending_list = os.listdir(config['dcmfile_path'])
    res_inference = inference(models, prending_list)

    print('------------------- inference completed at {}-------------------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))

if __name__ == '__main__':
    main()