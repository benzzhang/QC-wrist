'''
Date: 2023-04-21 10:52:12
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-09-22 16:11:34
FilePath: /QC-wrist/train_landmark.py
Description: Copyright (c) Pengbo, 2022
            Landmarks detection model, using DATASET 'WristLandmarkMaskDataset'
'''

import os
import time
import yaml
import numpy as np

import torch
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data.dataloader import default_collate
from matplotlib import pyplot as plt  

import models
import dataset
from utils import Logger, AverageMeter, mkdir_p, save_checkpoint, progress_bar, visualize_heatmap, get_landmarks_from_heatmap
import losses
import cv2
import pydicom
import math


def main(config_file):
    # parse config of model training
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    common_config = config['common']

    mkdir_p(common_config['save_path'])

    # initial dataset and dataloader
    augment_config = config['augmentation']
    
    global data_config
    data_config = config['dataset']
    print('==> Preparing dataset %s' % data_config['type'])
    # create dataset for training and validating
    if 'LAT' in common_config['project']:
        merge = True
    merge = False
    trainset = dataset.__dict__[data_config['type']](
        data_config['train_list'], data_config['train_meta'], augment_config,
        prefix=data_config['prefix'], size=(data_config['W_size'], data_config['H_size']), merge=merge)
    validset = dataset.__dict__[data_config['type']](
        data_config['valid_list'], data_config['valid_meta'], {'rotate_angle': 0, 'offset': [0, 0]},
        prefix=data_config['prefix'], size=(data_config['W_size'], data_config['H_size']), merge=merge)

    # create dataloader for training and validating
    '''
        kepp the image names in DATALOADER
    '''
    def name_collate(batch):
        new_batch = []
        names = []
        for _batch in batch:
            new_batch.append(_batch[:-1])
            names.append(_batch[-1])
        return default_collate(new_batch), names

    trainloader = data.DataLoader(
        trainset, batch_size=common_config['train_batch'], shuffle=True, num_workers=5)
    validloader = data.DataLoader(
        validset, batch_size=common_config['valid_batch'], shuffle=False, num_workers=5, collate_fn=name_collate)

    # Model
    print("==> creating model '{}'".format(common_config['arch']))
    model = models.__dict__[common_config['arch']](
        num_classes=data_config['num_classes'], local_net=common_config['local_net'])
    model = torch.nn.DataParallel(model)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    from torchsummary import summary
    # summary(model, (3, 960, 1920))

    # loss, optimizer and scheduler
    criterion = losses.__dict__[config['loss_config']['type']]()
    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=common_config['lr'],
        weight_decay=common_config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **common_config[common_config['scheduler_lr']])

    if args.visualize:
        checkpoints = torch.load(os.path.join('checkpoints/', 'model_best_{}.pth.tar'.format(common_config['project'])))
        model.load_state_dict(checkpoints['state_dict'], False)
        _, radial_errors = valid(validloader, model, criterion, use_cuda, common_config, visualize=args.visualize)

        def flat(nums):
            res = []
            for i in nums:
                if isinstance(i, list):
                    res.extend(flat(i))
                else:
                    res.append(i)
            return res
        
        radial_error = flat(radial_errors)

        # percentile
        p = [50, 80, 85, 90, 95]
        percentile = np.percentile(radial_error, p)

        mre = np.mean(np.array(radial_error))
        mre_sd = np.std(np.array(radial_error))

        SDR_1_0mm = len([i for i in radial_error if i <= 1.]) / len(radial_error)
        SDR_2_0mm = len([i for i in radial_error if i <= 2.]) / len(radial_error)
        SDR_2_5mm = len([i for i in radial_error if i <= 2.5]) / len(radial_error)
        SDR_3_0mm = len([i for i in radial_error if i <= 3.]) / len(radial_error)
        SDR_4_0mm = len([i for i in radial_error if i <= 4.]) / len(radial_error)
        SDR_10_0mm = len([i for i in radial_error if i <= 10.]) / len(radial_error)

        # indicators for each point
        SDR_4_0mm_mul = []
        for n in radial_errors:
            SDR_4_0mm_mul.append(len([i for i in n if i <= 4.]) / len(n))

        indicators_path = os.path.join('experiments/', common_config['project'], 'indicators_of_valid.txt')
        with open(indicators_path, 'a') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
            f.write('MRE(SD): %.4f ± %.4f' %(mre, mre_sd) + '\n')
            f.write('SDR_1.0mm: %.4f' %(SDR_1_0mm) + '\n')
            f.write('SDR_2.0mm: %.4f' %(SDR_2_0mm) + '\n')
            f.write('SDR_2.5mm: %.4f' %(SDR_2_5mm) + '\n')
            f.write('SDR_3.0mm: %.4f' %(SDR_3_0mm) + '\n')
            f.write('SDR_4.0mm: %.4f' %(SDR_4_0mm) + '\n')
            for idx in range(len(SDR_4_0mm_mul)):
                f.write('   SDR_4.0mm for P%d: %.4f' %(idx+1, SDR_4_0mm_mul[idx]) + '\n')
            f.write('SDR_10.0mm: %.4f' %(SDR_10_0mm) + '\n')
            # f.write('percentile %d%%: %.4f' %(p[0], percentile[0]) + '\n')
            # f.write('percentile %d%%: %.4f' %(p[1], percentile[1]) + '\n')
            # f.write('percentile %d%%: %.4f' %(p[2], percentile[2]) + '\n')
            # f.write('percentile %d%%: %.4f' %(p[3], percentile[3]) + '\n')
            # f.write('percentile %d%%: %.4f' %(p[4], percentile[4]) + '\n')
            f.write('━━●●━━━━━━━━━━━━━' + '\n')

        return

    # logger
    logger_path = os.path.join('experiments/', common_config['project'])
    mkdir_p(logger_path)
    title = 'Wrist landamrks detection using' + common_config['arch']
    logger = Logger(os.path.join(logger_path, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Avg-Train Loss', 'Avg-Valid Loss'])

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler(enabled=True) if config['common']['fp16'] == True else None
    best_loss = float('inf')
    # Train and val
    train_loss_list, valid_loss_list = [], []
    for epoch in range(common_config['epoch']):
        lr = scheduler.get_last_lr()[0]
        print('\nEpoch: [%d | %d] LR: %f' %(epoch + 1, common_config['epoch'], lr))
        train_loss = train(trainloader, model, criterion, optimizer, use_cuda, scaler, scheduler)
        valid_loss, _ = valid(validloader, model, criterion, use_cuda, common_config, scaler=scaler)
        scheduler.step()
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)

        # append logger file & save model
        logger.append([lr, train_loss, valid_loss])
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, save_path='checkpoints/', 
        ckp_name='checkpoint_{}.pth.tar'.format(common_config['project']),
        best_name='model_best_{}.pth.tar'.format(common_config['project']))
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

    def draw(train, valid):
        x = np.linspace(0,len(train),len(valid))  
        plt.plot(x,train,label="train_loss",linewidth=1.5)  
        plt.plot(x,valid,label="test_loss",linewidth=1.5)  
        plt.xlabel("epoch")  
        plt.ylabel("loss")  
        plt.legend()  
        plt.savefig(os.path.join(common_config['save_path'], 'loss_'+time.strftime('%Y-%m-%d_%H:%M:%S')+'.png'),
                     dpi=400,
                     bbox_inches='tight')
        
    draw(train_loss_list, valid_loss_list)
    print('Best loss:' + str(best_loss))
    logger.close(best_loss)

def train(trainloader, model, criterion, optimizer, use_cuda, scaler=None, scheduler=None):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for batch_idx, datas in enumerate(trainloader):
        if len(datas) == 4:
            inputs, targets, masks, _ = datas
            if use_cuda:
                masks = masks.cuda()
            masks = torch.autograd.Variable(masks)
        else:
            inputs, targets = datas
            masks = None
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = torch.autograd.Variable(inputs)
        targets = torch.autograd.Variable(targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if scaler is None:
            outputs = model(inputs)
            loss = criterion(outputs, targets, masks) / (outputs.size(0) * outputs.size(1))
            loss.backward()
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets, masks) / (outputs.size(0) * outputs.size(1))
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(loss).backward()
            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()

        losses.update(loss.item(), inputs.size(0))
        progress_bar(batch_idx, len(trainloader), 'Loss: %.2f' % (losses.avg))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg


def valid(validloader, model, criterion, use_cuda, common_config, scaler=None, visualize=None):
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    landmarks_list = []
    names_list = []
    radial_error = []
    for batch_idx, datas in enumerate(validloader): 
        (inputs, targets, masks), names = datas # because of 'collate_fn'
        if use_cuda:
            inputs, targets, masks = inputs.cuda(), targets.cuda(), masks.cuda()

        inputs = torch.autograd.Variable(inputs)
        targets = torch.autograd.Variable(targets)
        masks = torch.autograd.Variable(masks)

        # compute output
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets, masks) / (outputs.size(0) * outputs.size(1))
        else:
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, targets, masks) / (outputs.size(0) * outputs.size(1))

        if visualize:
            save_folder = os.path.join(common_config['save_path'], 'visualized_results/')
            mkdir_p(save_folder)
            for i in range(inputs.size(0)):
                landmarks = get_landmarks_from_heatmap(outputs[i].detach(), project=common_config['project'])
                # calculate 'mean radial error' (MRE) and 'successful detection rates' (SDR)
                landmarks_gt = get_landmarks_from_heatmap(targets[i].detach(), project=common_config['project'])

                visualize_img = visualize_heatmap(inputs[i], landmarks, landmarks_gt)
                save_path = os.path.join(save_folder, names[i])
                cv2.imwrite(save_path, visualize_img)
                landmarks_list.append(landmarks)
                names_list.append(names[i])

                if len(radial_error) == 0:
                    for n in range(len(landmarks)):
                        radial_error.append([])
                for idx, ([y, x], [y_gt, x_gt]) in enumerate(zip(landmarks, landmarks_gt)):
                    posture = args.config_file.split('_')[-1].split('.')[0]
                    # this path stored the original 'dcm' file, and read them just for obtaining the PixelSpacing
                    if y_gt==0. and x_gt==0.:
                        continue
                    dcmfile = os.path.join('/data/experiments/wrist_data_dcm/wrist_'+posture, names[i].replace('png', 'dcm'))
                    df = pydicom.read_file(dcmfile, force=True)

                    if not hasattr(df.file_meta, 'TransferSyntaxUID'):
                        df.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                    size = df.pixel_array.shape
                    PixelSpacing = df.data_element('PixelSpacing').value
                    PixelSpacing = (float(PixelSpacing._list[0]), float(PixelSpacing._list[1]))
                    PixelSpacing = [PixelSpacing[0] * (size[0] / data_config['H_size']), PixelSpacing[1] * (size[1] / data_config['W_size'])]

                    # unit: mm
                    r = math.sqrt(((y - y_gt) * PixelSpacing[0])**2  + ((x - x_gt) * PixelSpacing[1])**2)
                    radial_error[idx].append(r)

                    if r > 20:
                        # print('%s P%d %dmm' %(names[i], idx+1,np.floor(r)))
                        pass

                landmarks_array = np.array(landmarks_list).reshape(len(landmarks_list), -1)
                position_path = os.path.join('experiments/', common_config['project'], 'pred_landmarks.txt')
                filenames_path = os.path.join('experiments/', common_config['project'], 'pred_filenames.txt')

                np.savetxt(position_path, landmarks_array, fmt='%.2f')
                # with open(save_path_pos, 'r+') as f:
                #     content = f.read()
                #     f.seek(0, 0)
                #     f.write('cases number:' + str(landmarks_array.shape[0]) + '\n' + content)
                with open(filenames_path, 'w+') as f:
                    for i in names_list:
                        f.write(i+'\n')

        losses.update(loss.item(), inputs.size(0))
        progress_bar(batch_idx, len(validloader), 'Loss: %.2f' % (losses.avg))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, radial_error

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Landmark Detection for Medical Image')
    # model related, including  Architecture, path, datasets
    # parser.add_argument('--config-file', type=str, default='configs/config_landmarks_AP.yaml')
    parser.add_argument('--config-file', type=str,default='configs/config_landmarks_LAT.yaml')
    parser.add_argument('--gpu-id', type=str, default='0,1,2')
    parser.add_argument('--visualize', action='store_false')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main(args.config_file)