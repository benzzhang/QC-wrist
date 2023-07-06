'''
Date: 2023-07-04 17:12:56
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-07-06 17:58:39
FilePath: /QC-wrist/depoly/torch2onnx.py
Description: 
'''
import torch
import onnx
import onnxruntime
import yaml
import os
import sys
sys.path.append('..')
import models
import time
import numpy as np
print(os.getcwd())


def pytorch2onnx(model, dummy_input):
    onnx_model_path = './onnx_model_' + time.strftime('%Y%m%d%H%M') + '.onnx'
    torch.onnx.export(
        model.module,
        dummy_input, 
        onnx_model_path, 
        # verbose=True,
        export_params=True,
        input_names=['input'],
        output_names=['output'],
        training=torch.onnx.TrainingMode.EVAL,
        dynamic_axes={
            "input":{0: "batch_size"},
            "output":{0: "batch_size"}
            })

    return onnx_model_path


def onnx_check(model_path):
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    # print(onnx.helper.printable_graph(onnx_model.graph))


def onnx_inference(model_path, dummy_input, providers):
    session = onnxruntime.InferenceSession(model_path, providers=providers)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    t0 = time.time()
    outputs = session.run([], {input_name: dummy_input.cpu().numpy()})
    t1 = time.time()

    return outputs, (t1-t0)*1000

if __name__ == '__main__':
    global config
    
    with open('../experiments/config_inference.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_classify = models.__dict__[config['arch_classify']](num_classes=config['num_classes_classify'])
    model_landmark_AP = models.__dict__[config['arch_lanmarks']](num_classes=config['num_classes_AP'], local_net=config['local_net'])
    model_landmark_LAT = models.__dict__[config['arch_lanmarks']](num_classes=config['num_classes_LAT'], local_net=config['local_net'])

    model_classify = torch.nn.DataParallel(model_classify)
    model_landmark_AP = torch.nn.DataParallel(model_landmark_AP)
    model_landmark_LAT = torch.nn.DataParallel(model_landmark_LAT)

    model_classify.load_state_dict(torch.load(os.path.join(config['save_path_classify'], 'model_best.pth.tar'))['state_dict'], strict=True)
    model_landmark_AP.load_state_dict(torch.load(os.path.join(config['save_path_AP'], 'model_best.pth.tar'))['state_dict'], strict=True)
    model_landmark_LAT.load_state_dict(torch.load(os.path.join(config['save_path_LAT'], 'model_best.pth.tar'))['state_dict'], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_classify = model_classify.to(device)
    model_landmark_AP = model_landmark_AP.to(device)
    model_landmark_LAT = model_landmark_LAT.to(device)

    model_classify.eval()
    model_landmark_AP.eval()
    model_landmark_LAT.eval()

    dummy_input = torch.randn(3, 3, config['size_classify']['W'], config['size_classify']['H'], dtype=torch.float32)
    dummy_input = dummy_input.to(device)
    
    onnx_model_path = pytorch2onnx(model_classify, dummy_input)
    onnx_check(onnx_model_path)

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    result_onnx, time_onnx = onnx_inference(onnx_model_path, dummy_input, providers)
    print(result_onnx)

    t0 = time.time()
    result_pytorch = model_classify(dummy_input)
    t1 = time.time()

    print(result_pytorch)
    print(np.abs(result_onnx - result_pytorch.detach().cpu().numpy()))
    print('ONNX cost {:.2f}ms'.format(time_onnx), 'Pytorch cost {:.2f}ms'.format((t1-t0)*1000))
