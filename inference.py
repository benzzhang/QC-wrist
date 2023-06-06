'''
Date: 2023-05-26 10:19:09
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-06-06 10:50:08
FilePath: /QC-wrist/inference.py
Description: 
'''

import os
import yaml
import os
import time
import yaml
import numpy as np
import pydicom
import torch
import cv2

import models
from utils import get_landmarks_from_heatmap, visualize_heatmap
from eval import is_position_mark, flip_AP, flip_LAT, midpoint_of_StyloidProcess_is_center, line_of_LongAxis_is_vertical,\
include_radius_ulna, distance_from_StyloidProcess_to_edge, Scaphoid_is_center, line_of_StyloidProcess_is_horizontal,\
basic_information_completed, dose, radius_and_ulna_overlap, distal_radius_and_ulna_overlap


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
    model_classify = torch.nn.DataParallel(model_classify)
    model_landmark_AP = torch.nn.DataParallel(model_landmark_AP)
    model_landmark_LAT = torch.nn.DataParallel(model_landmark_LAT)
    model_classify.load_state_dict(torch.load(os.path.join(config['save_path_classify'], 'model_best.pth.tar'))['state_dict'], strict=True)
    model_landmark_AP.load_state_dict(torch.load(os.path.join(config['save_path_AP'], 'model_best.pth.tar'))['state_dict'], strict=True)
    model_landmark_LAT.load_state_dict(torch.load(os.path.join(config['save_path_LAT'], 'model_best.pth.tar'))['state_dict'], strict=True)

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

        if np.max(resized_df1) > 1:
            resized_df1 = (resized_df1 - 127.5) / 127.5
        else:
            resized_df1 = (resized_df1 - 0.5) * 2

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

        '''
            generate image which keeping original size with landmarks
        '''
        size = df_pixel.shape
        actual_coordinate = []
        for c in result[2]:
            ac = [(c[0]/config['size_landmarks']['H'])*size[0], (c[1]/config['size_landmarks']['W'])*size[1]]
            ac = [int(ac[0]), int(ac[1])]
            actual_coordinate.append(ac)
            
        img = visualize_heatmap(input=cv2.merge([scaled_df_pixel, scaled_df_pixel, scaled_df_pixel]),
                                landmarks=actual_coordinate)
        cv2.imwrite(os.path.join('./inference_result', i.replace('dcm', 'png')), img)

    return res_list

def evaluate_each(dcmfile, coordinate, score_dict):
    df = pydicom.read_file(dcmfile, force=True)
    ProtocolName = df.data_element('ProtocolName').value
    
    df.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    size = df.pixel_array.shape

    '''
        calculate
    '''
    PixelSpacing = df.data_element('PixelSpacing').value
    PixelSpacing = [float(PixelSpacing._list[0]),float(PixelSpacing._list[1])]
    PixelSpacing = [PixelSpacing[0] * (size[0] / config['size_landmarks']['H']), PixelSpacing[1] * (size[1] / config['size_landmarks']['W'])]

    '''
        PixelSpacing: (H, W) = (y, x)
        coordinate: (H, W) = (y, x)
        size: (H, W) = (y, x)
        'needed to reverse them'
    '''
    size = [size[1], size[0]]
    PixelSpacing = [PixelSpacing[1], PixelSpacing[0]]
    for idx, point in enumerate(coordinate):
        coordinate[idx] = [point[1], point[0]]

    if ProtocolName == '腕关节正位' or len(coordinate) == 3:
        p1 = coordinate[0]
        p2 = coordinate[1]
        p3 = coordinate[2]

        p1, p2, p3, size = flip_AP(p1, p2, p3, size)

        s1 = midpoint_of_StyloidProcess_is_center(p2, p3, PixelSpacing, size)
        s2 = line_of_StyloidProcess_is_horizontal(p2, p3)
        s3 = include_radius_ulna(p2, p3, PixelSpacing, size)
        s4 = distance_from_StyloidProcess_to_edge(p2, p3, PixelSpacing, size)
        layout_score = s1 + s2 + s3 + s4
        
        score_dict['尺桡骨茎突连线中点位于图像正中'] = s1
        score_dict['尺桡骨茎突连线与图像纵轴垂直'] = s2
        score_dict['下缘包含尺桡骨3-5cm'] = s3
        score_dict['左右最外侧距影像边缘3-5cm'] = s4
        
    if ProtocolName == '腕关节侧位' or len(coordinate) == 5:
        p1 = coordinate[0]
        p2 = coordinate[1]
        p3 = coordinate[2]
        p4 = coordinate[3]
        p5 = coordinate[4]

        p1, p2, p3, p4, p5, size = flip_LAT(p1, p2, p3, p4, p5, size)

        s1 = Scaphoid_is_center(p1, PixelSpacing, size)
        s2 = line_of_LongAxis_is_vertical(p1, p3, p5)
        s3 = radius_and_ulna_overlap(p2, p3, p4, p5)
        s4 = distal_radius_and_ulna_overlap(p1, p2, p4)
        layout_score = s1 + s2 + s3 + s4

        score_dict['舟骨位于图像正中'] = s1
        score_dict['腕关节长轴与影像长轴平行'] = s2

    score_basic = basic_information_completed(df)
    score_dose = dose(df)
    layout_score = layout_score + score_basic + score_dose

    score_dict['基本信息完整度'] = score_basic
    score_dict['辐射剂量'] = score_dose

    return layout_score, score_dict


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
        Obtain a list of files to be inferred
    '''
    prending_list = [f for f in os.listdir(config['dcmfile_path']) if not f.startswith('.')]
    res_inference = inference(models, prending_list)

    res_dict = dict()
    for res, path in zip(res_inference, prending_list):
        score_dict = dict()

        qualified_flag = True
        dcmfile = os.path.join(config['dcmfile_path'], path)
        if not res[0]:
            qualified_flag = False
        score_dict['mark'] = qualified_flag

        if res[1]:
            score = 5
        else:
            score = 10
        score_dict['artifact'] = qualified_flag

        layout_score, score_dict = evaluate_each(dcmfile, res[2], score_dict)
        score += layout_score

        if qualified_flag:
            res_dict[str(path)] = score
        else:
            res_dict[str(path)] = 0

        res_dict['detail of '+str(path)] = score_dict
    
    import json
    json_str = json.dumps(res_dict, ensure_ascii=False)
    f = open('inference_result/inference.json', 'w')
    f.write(json_str)
    f.close()
    
    print('------------------- inference completed at {}-------------------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))

if __name__ == '__main__':
    main()