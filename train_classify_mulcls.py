'''
Date: 2023-04-21 10:52:11
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-05-23 17:42:23
FilePath: /QC-wrist/train_classify_mulcls.py
Description: 
'''
import os
import shutil
import time

import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import yaml

from augmentation.medical_augment import XrayTrainTransform

import models
import dataset
import losses
from utils import Logger, AverageMeter, mkdir_p, progress_bar

def main(config_file):
    global common_config, best_acc
    best_acc = 0
    # parse config of model training
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    common_config = config['common']

    if not os.path.isdir(common_config['save_path']):
        mkdir_p(common_config['save_path'])

    # initial dataset and dataloader
    data_config = config['dataset']
    print('==> Preparing dataset %s' % data_config['type'])
    # transform_train = XrayTrainTransform(crop_size=data_config['crop_size'], img_size=data_config['img_size'])
    transform_train = XrayTrainTransform()

    # create dataset for training and validating
    trainset = dataset.__dict__[data_config['type']](
        data_config['train_list'], data_config['train_meta'], transform_train,
        prefix=data_config['prefix'], size=(data_config['W_size'], data_config['H_size']))
    validset = dataset.__dict__[data_config['type']](
        data_config['valid_list'], data_config['valid_meta'], None,
        prefix=data_config['prefix'], size=(data_config['W_size'], data_config['H_size']))

    # create dataloader for training and validating

    '''
        kepp all images having same size in ONE BATCH
    '''
    def collate_syn(batch):
        sizes_w = [item[0].shape[1] for item in batch]
        sizes_h = [item[0].shape[0] for item in batch]
        # print(sizes_w, sizes_h)
        max_w, max_h = max(sizes_w), max(sizes_h)
        # print('max size:%d %d' % (max_w, max_h))

        packaged_images = []
        packaged_labels = []
        for item in batch:
            img = cv2.resize(item[0], (max_w, max_h))
            img = torch.FloatTensor(img.transpose((2, 0, 1)))
            packaged_images.append(img)
            packaged_labels.append(item[1])
        
        tensor1 = torch.stack(packaged_images)
        tensor2 = torch.stack(packaged_labels)
        return tensor1, tensor2

    trainloader = data.DataLoader(
        trainset, batch_size=common_config['train_batch'], shuffle=True, num_workers=5)
    validloader = data.DataLoader(
        validset, batch_size=common_config['valid_batch'], shuffle=False, num_workers=5)

    # Model
    print("==> creating model '{}'".format(common_config['arch']))
    model = models.__dict__[common_config['arch']](
        n_foreign_classes=data_config['foreign_classes'], n_position_classes=data_config['position_classes'])
    model = torch.nn.DataParallel(model)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    cudnn.benchmark = True

    from torchsummary import summary
    # summary(model, (3, 928, 2176))

    # loss & optimizer
    # criterion = losses.__dict__[config['loss']['type']]()
    criterion = torch.nn.BCELoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=common_config['lr'],
        momentum=common_config['momentum'],
        weight_decay=common_config['weight_decay'])

    # logger
    title = 'Wrist X-ray Image Quality Assessment using ' + common_config['arch']
    logger = Logger(os.path.join(common_config['save_path'], 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 
                      'Train Acc.', 'Valid Acc.',
                      'Train Acc_f.', 'Valid Acc_f.',
                      'Train Acc_p.', 'Valid Acc_p.'])

    # Train and val
    for epoch in range(common_config['epoch']):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, common_config['epoch'], common_config['lr']))
        train_loss, train_acc, train_acc_f, train_acc_p = train(trainloader, model, criterion, optimizer, use_cuda)
        valid_loss, valid_acc, valid_acc_f, valid_acc_p = vaild(validloader, model, criterion, use_cuda)
        # append logger file
        logger.append([common_config['lr'], train_loss, valid_loss, 
                       train_acc, valid_acc,
                       train_acc_f, valid_acc_f,
                       train_acc_p, valid_acc_p])
        # save model
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': valid_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, save_path=common_config['save_path'])

    print('Best acc:', best_acc)
    logger.close(best_acc)


def train(trainloader, model, criterion, optimizer, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_f = AverageMeter()
    acc_p = AverageMeter()
    end = time.time()

    for batch_idx, datas in enumerate(trainloader):
        inputs, f_targets, p_targets = datas

        if use_cuda:
            inputs, f_targets, p_targets = inputs.cuda(), f_targets.cuda(), p_targets.cuda()
        inputs, f_targets, p_targets = torch.autograd.Variable(inputs), torch.autograd.Variable(f_targets), torch.autograd.Variable(p_targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        outputs = model(inputs)

        loss1 = criterion(outputs['foreign'], f_targets)
        loss2 = criterion(outputs['position'], p_targets)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        predict_foreign = outputs['foreign'] > 0.5
        predict_position = outputs['position'] > 0.5

        concat_predict = torch.cat((predict_foreign, predict_position), 1)
        concat_target = torch.cat((f_targets, p_targets), 1)

        predict_res_f = [torch.equal(a, b) for a, b in zip(predict_foreign, f_targets)]
        predict_res_p = [torch.equal(a, b) for a, b in zip(predict_position, p_targets)]
        predict_res = [torch.equal(a, b) for a, b in zip(concat_predict, concat_target)]

        losses.update(loss.item(), inputs.size(0))
        acc_f.update(sum(predict_res_f) / len(predict_res_f), len(predict_res_f))
        acc_p.update(sum(predict_res_p) / len(predict_res_p), len(predict_res_p))
        acc.update(sum(predict_res) / len(predict_res), len(predict_res))

        progress_bar(batch_idx, len(trainloader), 'Loss: %.2f | Acc: %.2f | Acc_f: %.2f | Acc_p: %.2f' % (losses.avg, acc.avg, acc_f.avg, acc_p.avg))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, acc.avg, acc_f.avg, acc_p.avg)


def vaild(validloader, model, criterion, use_cuda):
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_f = AverageMeter()
    acc_p = AverageMeter()
    end = time.time()

    for batch_idx, datas in enumerate(validloader):
        inputs, f_targets, p_targets = datas

        if use_cuda:
            inputs, f_targets, p_targets = inputs.cuda(), f_targets.cuda(), p_targets.cuda()
        inputs, f_targets, p_targets = torch.autograd.Variable(inputs), torch.autograd.Variable(f_targets), torch.autograd.Variable(p_targets)

        # compute gradient and do SGD step
        outputs = model(inputs)
   
        loss1 = criterion(outputs['foreign'], f_targets)
        loss2 = criterion(outputs['position'], p_targets)
        loss = loss1 + loss2

        predict_foreign = outputs['foreign'] > 0.5
        predict_position = outputs['position'] > 0.5
        
        concat_predict = torch.cat((predict_foreign, predict_position), 1)
        concat_target = torch.cat((f_targets, p_targets), 1)

        predict_res_f = [torch.equal(a, b) for a, b in zip(predict_foreign, f_targets)]
        predict_res_p = [torch.equal(a, b) for a, b in zip(predict_position, p_targets)]
        predict_res = [torch.equal(a, b) for a, b in zip(concat_predict, concat_target)]

        losses.update(loss.item(), inputs.size(0))
        acc_f.update(sum(predict_res_f) / len(predict_res_f), len(predict_res_f))
        acc_p.update(sum(predict_res_p) / len(predict_res_p), len(predict_res_p))
        acc.update(sum(predict_res) / len(predict_res), len(predict_res))

        progress_bar(batch_idx, len(validloader), 'Loss: %.2f | Acc: %.2f | Acc_f: %.2f | Acc_p: %.2f' % (losses.avg, acc.avg, acc_f.avg, acc_p.avg))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, acc.avg, acc_f.avg, acc_p.avg)


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global common_config
    if epoch in common_config['scheduler']:
        common_config['lr'] *= common_config['gamma']
        for param_group in optimizer.param_groups:
            param_group['lr'] = common_config['lr']


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Classify for Medical Image')
    # model related, including  Architecture, path, datasets
    parser.add_argument('--config-file', type=str, default='experiments/config_classify_mul_AP.yaml')
    # parser.add_argument('--config-file', type=str, default='experiments/config_classify_LAT.yaml')
    parser.add_argument('--gpu-id', type=str, default='0,1,2')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main(args.config_file)
