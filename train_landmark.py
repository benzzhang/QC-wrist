'''
Date: 2023-04-21 10:52:12
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-05-24 16:57:18
FilePath: /QC-wrist/train_landmark.py
Description: Copyright (c) Pengbo, 2022
'''

import os
import shutil
import time
import yaml
import numpy as np

import torch
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data.dataloader import default_collate

import models
import dataset
from utils import Logger, AverageMeter, mkdir_p, progress_bar, visualize_heatmap, get_landmarks_from_heatmap
import losses
import cv2


def main(config_file):
    # parse config of model training
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    common_config = config['common']

    if not os.path.isdir(common_config['save_path']):
        mkdir_p(common_config['save_path'])

    # logger
    title = 'Wrist landamrks detection using' + common_config['arch']
    logger = Logger(os.path.join(
        common_config['save_path'], 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Avg-Train Loss', 'Avg-Valid Loss', 'Epoch-Train Loss', 'Epoch-Valid Loss'])

    # initial dataset and dataloader
    augment_config = config['augmentation']
    data_config = config['dataset']
    print('==> Preparing dataset %s' % data_config['type'])
    # create dataset for training and validating
    trainset = dataset.__dict__[data_config['type']](
        data_config['train_list'], data_config['train_meta'], augment_config,
        prefix=data_config['prefix'], size=(data_config['W_size'], data_config['H_size']))
    validset = dataset.__dict__[data_config['type']](
        data_config['valid_list'], data_config['valid_meta'], {'rotate_angle': 0, 'offset': [0, 0]},
        prefix=data_config['prefix'], size=(data_config['W_size'], data_config['H_size']))

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

    # optimizer and scheduler
    criterion = losses.__dict__[config['loss_config']['type']]()

    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=common_config['lr'],
        weight_decay=common_config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **common_config[common_config['scheduler_lr']])

    if args.visualize:
        checkpoints = torch.load(os.path.join(common_config['save_path'], 'model_best.pth.tar'))
        model.load_state_dict(checkpoints['state_dict'], False)
        _, _, landmarks_array, names = valid(validloader, model, criterion, use_cuda, common_config, visualize=args.visualize)

        save_folder = os.path.join(common_config['save_path'], 'results/')
        save_path_pos = os.path.join(save_folder, 'pred_landmarks.txt')
        save_path_names = os.path.join(save_folder, 'pred_img_names.txt')

        np.savetxt(save_path_pos, landmarks_array, fmt='%.2f')
        # with open(save_path_pos, 'r+') as f:
        #     content = f.read()
        #     f.seek(0, 0)
        #     f.write('cases number:' + str(landmarks_array.shape[0]) + '\n' + content)
        with open(save_path_names, 'w+') as f:
            for i in names:
                f.write(i+'\n')

        return

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler(enabled=True) if config['common']['fp16'] == True else None
    best_loss = float('inf')
    # Train and val
    for epoch in range(common_config['epoch']):
        lr = scheduler.get_last_lr()[0]
        print('\nEpoch: [%d | %d] LR: %f' %(epoch + 1, common_config['epoch'], lr))
        train_loss, ep_train_loss = train(trainloader, model, criterion, optimizer, use_cuda, scaler, scheduler)
        valid_loss, ep_valid_loss, _, _ = valid(validloader, model, criterion, use_cuda, common_config, scaler, args.visualize)
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)

        # append logger file & save model
        logger.append([lr, train_loss, valid_loss, ep_train_loss, ep_valid_loss])
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, save_path=common_config['save_path'])

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
        if scheduler != None:
            scheduler.step(batch_idx)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, loss.item()


def valid(validloader, model, criterion, use_cuda, common_config, scaler=None, visualize=None):
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    landmarks_list = []
    names_list = []

    for batch_idx, datas in enumerate(validloader):
        (inputs, targets, masks), names = datas
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
            save_folder = os.path.join(common_config['save_path'], 'results/')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            for i in range(inputs.size(0)):
                landmarks = get_landmarks_from_heatmap(outputs[i].detach())
                visualize_img = visualize_heatmap(inputs[i], landmarks)
                save_path = os.path.join(save_folder, str(batch_idx * inputs.size(0) + i) + '.png')
                cv2.imwrite(save_path, visualize_img)
                landmarks_list.append(landmarks)
                names_list.append(names[i])
        losses.update(loss.item(), inputs.size(0))
        progress_bar(batch_idx, len(validloader), 'Loss: %.2f' % (losses.avg))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if visualize:
        landmarks_array = np.array(landmarks_list).reshape(len(landmarks_list), -1)
        return losses.avg, loss.item(), landmarks_array, names_list
    else:
        return losses.avg, loss.item(), None, None


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_best.pth.tar'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Landmark Detection for Medical Image')
    # model related, including  Architecture, path, datasets
    parser.add_argument('--config-file', type=str, default='experiments/config_landmarks_AP.yaml')
    # parser.add_argument('--config-file', type=str,default='experiments/config_landmarks_LAT.yaml')
    parser.add_argument('--gpu-id', type=str, default='0,1,2')
    parser.add_argument('--visualize', action='store_false')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main(args.config_file)