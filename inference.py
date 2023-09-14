'''
Date: 2023-05-26 10:19:09
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-09-14 17:58:45
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
from utils import get_landmarks_from_heatmap, visualize_in_evaluate
from eval import is_position_mark, midpoint_of_StyloidProcess_is_center, line_of_LongAxis_is_vertical,\
include_radius_ulna, distance_from_StyloidProcess_to_edge, Scaphoid_is_center, line_of_StyloidProcess_is_horizontal,\
basic_information_completed, dose, radius_and_ulna_overlap, distal_radius_and_ulna_overlap, metacarpophalangeal_joint_is_included


def init_ai_model():
    '''
        initial the ai quality model
    '''
    print('Initializing the model at {} ...'.format(time.strftime("%Y-%m-%d %X", time.localtime())))

    global use_cuda
    model_classify_position = models.__dict__[config['arch']['classify']](num_classes=config['num_classes']['classify'])
    model_classify_artifact = models.__dict__[config['arch']['classify']](num_classes=config['num_classes']['classify'])
    model_classify_overlap = models.__dict__[config['arch']['classify']](num_classes=config['num_classes']['classify'])
    model_landmark_AP = models.__dict__[config['arch']['landmarks']['net']](num_classes=config['num_classes']['AP'], local_net=config['arch']['landmarks']['local_net'])
    model_landmark_LAT = models.__dict__[config['arch']['landmarks']['net']](num_classes=config['num_classes']['LAT'], local_net=config['arch']['landmarks']['local_net'])

    # 'torch.nn.DataParallel' will load model on the default CUDA
    model_classify_position = torch.nn.DataParallel(model_classify_position)
    model_classify_artifact = torch.nn.DataParallel(model_classify_artifact)
    model_classify_overlap = torch.nn.DataParallel(model_classify_overlap)
    model_landmark_AP = torch.nn.DataParallel(model_landmark_AP)
    model_landmark_LAT = torch.nn.DataParallel(model_landmark_LAT)

    # 'load_state_dict' starting to occupy the memory of GPU
    model_classify_position.load_state_dict(torch.load(os.path.join('checkpoints/', config['checkpoints']['classify_position']))['state_dict'], strict=True)
    model_classify_artifact.load_state_dict(torch.load(os.path.join('checkpoints/', config['checkpoints']['classify_artifact']))['state_dict'], strict=True)
    model_classify_overlap.load_state_dict(torch.load(os.path.join('checkpoints/', config['checkpoints']['classify_overlap']))['state_dict'], strict=True)
    model_landmark_AP.load_state_dict(torch.load(os.path.join('checkpoints/', config['checkpoints']['landmarks_AP']))['state_dict'], strict=True)
    model_landmark_LAT.load_state_dict(torch.load(os.path.join('checkpoints/', config['checkpoints']['landmarks_LAT']))['state_dict'], strict=True)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model_classify_position = model_classify_position.cuda()
        model_classify_artifact = model_classify_artifact.cuda()
        model_classify_overlap = model_classify_overlap.cuda()
        model_landmark_AP = model_landmark_AP.cuda()
        model_landmark_LAT = model_landmark_LAT.cuda()
    return model_classify_position, model_classify_artifact, model_classify_overlap, model_landmark_AP, model_landmark_LAT


def inference(models, prending_list):
    '''
        evaluate the dicom by check tags completation and ai_quality_model, return the detail scores
    '''
    for model in models:
        model.eval()

    print('start inferencing at {} ...'.format(time.strftime("%Y-%m-%d %X", time.localtime())))
    res_list = []
    for i in prending_list:
        dcmfile = os.path.join(config['dcmfile_path'], i)
        df = pydicom.read_file(dcmfile, force=True)

        df.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        df_pixel = df.pixel_array
        scaled_df_pixel = (df_pixel - min(df_pixel.flatten())) / (max(df_pixel.flatten()) - min(df_pixel.flatten()))
        scaled_df_pixel *= 255
        resized_df0 = cv2.resize(scaled_df_pixel, (config['input_size']['classify']['W'], config['input_size']['classify']['H']))
        resized_df1 = cv2.resize(scaled_df_pixel, (config['input_size']['landmarks']['W'], config['input_size']['landmarks']['H']))

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

        try:
            ProtocolName = df.data_element('ProtocolName').value
        except:
            ProtocolName = None

        # position mark detection
        res_mark = is_position_mark(scaled_df_pixel)

        # model inferring
        with torch.no_grad():
        # classify for position and artifact
            res_classify_position = models[0](df_tensor0)
            res_classify_artifact = models[1](df_tensor0)
            # judge position & generate landmark
            if ProtocolName == '腕关节正位':
                flag = 'wrist-landmarks-AP'
                res_landmark = models[3](df_tensor1)
            elif ProtocolName == '腕关节侧位':
                flag = 'wrist-landmarks-LAT'
                # in LAT, an classify for overlap
                res_classify_overlap = models[2](df_tensor0)
                res_landmark = models[4](df_tensor1)

            else:
                if res_classify_position[0][0].item() > res_classify_position[0][1].item():
                    flag = 'wrist-landmarks-AP'
                    res_landmark = models[3](df_tensor1)
                else:
                    flag = 'wrist-landmarks-LAT'
                    res_classify_overlap = models[2](df_tensor0)
                    res_landmark = models[4](df_tensor1)
        # release the cache of PyTorch&CUDA
        # torch.cuda.empty_cache()
        '''
            return: True/False, True/False, List, True/False/None
        '''
        result = (res_mark, 
                  res_classify_artifact[0][0].item() > res_classify_artifact[0][1].item(), 
                  get_landmarks_from_heatmap(res_landmark.squeeze().detach(), project=flag),
                  res_classify_overlap[0][0].item() > res_classify_overlap[0][1].item() if 'LAT' in flag else None)
        res_list.append(result)

        '''
            generate image which keeping original size with landmarks
        '''
        size = df_pixel.shape
        actual_coordinate = []
        for c in result[2]:
            ac = [(c[0]/config['input_size']['landmarks']['H'])*size[0], (c[1]/config['input_size']['landmarks']['W'])*size[1]]
            ac = [int(ac[0]), int(ac[1])]
            actual_coordinate.append(ac)

        PixelSpacing = df.data_element('PixelSpacing').value
        PixelSpacing = [float(PixelSpacing._list[0]), float(PixelSpacing._list[1])]
        img = visualize_in_evaluate(input=cv2.merge([scaled_df_pixel, scaled_df_pixel, scaled_df_pixel]),
                                        landmarks=actual_coordinate,
                                        pixelspacing=PixelSpacing)
        cv2.imwrite(os.path.join(config['save_path'], i.replace('dcm', 'png')), img)

    return res_list

def evaluate_each(dcmfile, coordinate, overlap, score_dict):
    df = pydicom.read_file(dcmfile, force=True)
    
    df.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    size = df.pixel_array.shape

    '''
        calculate
    '''
    # This PixelSpacing is a 'relative PixelSpacing' in the resized img
    PixelSpacing = df.data_element('PixelSpacing').value
    PixelSpacing = [float(PixelSpacing._list[0]), float(PixelSpacing._list[1])]

    actual_coordinate = []
    for c in coordinate:
        ac = [(c[0]/config['input_size']['landmarks']['H'])*size[0], (c[1]/config['input_size']['landmarks']['W'])*size[1]]
        ac = [int(ac[0]), int(ac[1])]
        actual_coordinate.append(ac)

    '''
        PixelSpacing: (H, W) = (y, x)
        coordinate: (H, W) = (y, x)
        size: (H, W) = (y, x)
    '''
    # AP
    if overlap is None:
        p1 = actual_coordinate[0]
        p2 = actual_coordinate[1]
        p3 = actual_coordinate[2]

        s1 = metacarpophalangeal_joint_is_included(p1)
        s2, gap0, gap1 = midpoint_of_StyloidProcess_is_center(p2, p3, PixelSpacing, size)
        s3, angle_yaxis = line_of_StyloidProcess_is_horizontal(p2, p3, size)
        s4, distance_from_lowest = include_radius_ulna(p1, p2, p3, PixelSpacing, size)
        s5, distance_l, distance_r = distance_from_StyloidProcess_to_edge(p2, p3, PixelSpacing, size)
        layout_score = s1 + s2 + s3 + s4 + s5
        
        score_dict['上缘包含拇指指掌关节'] = s1
        score_dict['尺桡骨茎突连线中点位于图像正中'] = {'score': s2, '轴1方向差值': gap0, '轴2方向差值': gap1}
        score_dict['尺桡骨茎突连线与影像纵轴垂直'] = {'score': s3, '纵轴角度': angle_yaxis}
        score_dict['下缘包含尺桡骨3-5cm'] = {'score': s4, '尺桡骨长度': distance_from_lowest}
        score_dict['左右最外侧距影像边缘3-5cm'] = {'score': s5, '左侧茎突距离': distance_l, '右侧茎突距离': distance_r}
        
    # LAT
    else:
        p1 = actual_coordinate[0]
        p2 = actual_coordinate[1]
        p3 = actual_coordinate[2]
        p4 = actual_coordinate[3]
        p5 = actual_coordinate[4]

        s1, gap0, gap1 = Scaphoid_is_center(p1, PixelSpacing, size)
        s2, angle_yaxis = line_of_LongAxis_is_vertical(p1, p3, p5, size)
        s3 = 9 if overlap else 0
        # s3 = radius_and_ulna_overlap(p2, p3, p4, p5)
        s4, angleP1 = distal_radius_and_ulna_overlap(p1, p2, p4)
        layout_score = s1 + s2 + s3 + s4

        score_dict['舟骨位于图像正中'] = {'score': s1, '轴1方向差值': gap0, '轴2方向差值': gap1}
        score_dict['腕关节长轴与影像纵轴平行'] = {'score': s2, '纵轴角度': angle_yaxis}
        score_dict['尺桡骨重叠'] = True if s3==0 else False
        score_dict['尺桡骨远端重叠'] = {'score': s4, '远端夹角角度': angleP1}

    score_basic = basic_information_completed(df)
    score_dose = dose(df)
    dcm_score = layout_score + score_basic + 0

    score_dict['基本信息完整度'] = score_basic
    score_dict['辐射剂量(<5mGy)'] = score_dose

    return dcm_score, score_dict


def main():
    import argparse
    global config
    
    parser = argparse.ArgumentParser(description='workflow of QC in wrist')
    # model related, including  Architecture, path, datasets
    parser.add_argument('--config-file', type=str, default='configs/config_inference.yaml')
    parser.add_argument('--gpu-id', type=str, default='0')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    models = init_ai_model()

    '''
        Obtain a list of files to be inferred
    '''
    prending_list = [f for f in os.listdir(config['dcmfile_path']) if not f.startswith('.')]
    res_inference = inference(models, prending_list)

    print('scoring each image starting from {} ...'.format(time.strftime("%Y-%m-%d %X", time.localtime())))
    res_dict = dict()
    for res, path in zip(res_inference, prending_list):
        score_dict = dict()
        score = 0
        
        dcmfile = os.path.join(config['dcmfile_path'], path)

        score_dict['左右标识'] = res[0]
        if res[0]:
            score += 20

        score_dict['异物伪影'] = res[1]
        if res[1]:
            score += 0
        else:
            score += 10

        dcm_score, score_dict = evaluate_each(dcmfile, res[2], res[3], score_dict)
        score += dcm_score

        if res[0]:
            res_dict[str(path)] = score
        else:
            res_dict[str(path)] = 0

        res_dict['details of '+str(path)] = score_dict
    
    import json
    json_str = json.dumps(res_dict, ensure_ascii=False)
    f = open('inference_result/inference.json', 'w')
    f.write(json_str)
    f.close()
    
    print('{} cases have been inferred and scored, completed at {}'.format(len(prending_list), time.strftime("%Y-%m-%d %X", time.localtime())))

if __name__ == '__main__':
    main()