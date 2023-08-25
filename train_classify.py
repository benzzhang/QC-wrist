'''
@Author     : Jian Zhang
@Init Date  : 2023-05-17 13:57
@File       : train_classify.py
@IDE        : PyCharm
@Description: Binary classification model, using DATASET 'XrayClassifyDataset'
'''
import os
import time

import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import yaml
import numpy as np
import random

import sklearn
from augmentation.medical_augment import XrayTrainTransform

import models
import dataset
import sklearn
import losses
from utils import Logger, AverageMeter, mkdir_p, progress_bar, save_checkpoint

def main(config_file):
    global common_config, best_acc
    best_acc = 0
    # parse config of model training
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    common_config = config['common']

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
    model = models.__dict__[common_config['arch']](num_classes=data_config['num_classes'])
    model = torch.nn.DataParallel(model)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    cudnn.benchmark = True

    from torchsummary import summary
    # summary(model, (3, 960, 1920))

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

    if args.visualize:
        checkpoints = torch.load(os.path.join('checkpoints/', 'model_best_{}.pth.tar'.format(common_config['project'])))
        model.load_state_dict(checkpoints['state_dict'], False)
        valid_loss, valid_acc, valid_sens, valid_spec, valid_prec, auc, ci_95 = vaild(validloader, model, criterion, use_cuda)

        save_folder = os.path.join(common_config['save_path'], 'visualized_results/')
        mkdir_p(save_folder)
        indicators_path = os.path.join('experiments/', common_config['project'], 'indicators_of_valid.txt')

        with open(indicators_path, 'a') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
            f.write('loss: %.4f' %(valid_loss) + '\n')
            f.write('accuracy: %.4f' %(valid_acc) + '\n')
            f.write('sensitivity: %.4f' %(valid_sens) + '\n')
            f.write('specificity: %.4f' %(valid_spec) + '\n')
            f.write('precision: %.4f' %(valid_prec) + '\n')
            f.write('F1: %.4f' %(2*valid_prec*valid_sens/(valid_prec+valid_sens)) + '\n')
            f.write('AUC:  %.4f(95%%CI: %4f~%4f)' %(auc, ci_95[0], ci_95[1]) + '\n')
            f.write('━━●●━━━━━━━━━━━━━' + '\n')

        #TODO: 分类结果写入 save_folder 目录中
        return

    # logger
    logger_path = os.path.join('experiments/', common_config['project'])
    mkdir_p(logger_path)
    title = 'Wrist X-ray Image Quality Assessment using ' + common_config['arch']
    logger = Logger(os.path.join(logger_path, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # Train and val
    for epoch in range(common_config['epoch']):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, common_config['epoch'], common_config['lr']))
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, use_cuda)
        valid_loss, valid_acc, _, _, _, _, _ = vaild(validloader, model, criterion, use_cuda)
        # valid_loss, valid_acc = vaild(validloader, model, criterion, use_cuda)
        # append logger file
        logger.append([common_config['lr'], train_loss, valid_loss, train_acc, valid_acc])
        # save model
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': valid_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, save_path='checkpoints/', 
        ckp_name='checkpoint_{}.pth.tar'.format(common_config['project']),
        best_name='model_best_{}.pth.tar'.format(common_config['project']))

    print('Best acc: %.4f' %(best_acc))
    logger.close(best_acc)

def train(trainloader, model, criterion, optimizer, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    end = time.time()

    for batch_idx, datas in enumerate(trainloader):
        inputs, targets = datas

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        outputs = model(inputs)

        outputs = outputs.view(outputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        predict = outputs > 0.5
        predict_res = [torch.equal(a, b) for a, b in zip(predict, targets)]

        losses.update(loss.item(), inputs.size(0))
        acc.update(sum(predict_res) / len(predict_res), len(predict_res))

        progress_bar(batch_idx, len(trainloader), 'Loss: %.2f | Acc: %.2f' % (losses.avg, acc.avg))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, acc.avg)


def vaild(validloader, model, criterion, use_cuda):
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()    # accuracy:    (TP+TN)/(TP+TN+FP+FN)
    sens = AverageMeter()   # sensitivity: TP/(TP+FN) <==> recall
    spec = AverageMeter()   # specificity: TP/(FP+TN)
    prec = AverageMeter()   # precision:   TP/(TP+FP)
    end = time.time()

    labelList = []
    predList =[]
    for batch_idx, (inputs, targets) in enumerate(validloader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        
        with torch.no_grad():
            outputs = model(inputs)

        # calculating AUC, drawing ROC
        for i, j in zip(targets.tolist(), outputs.tolist()):
            labelList.append(i[0]) # 0-negative 1-positive
            predList.append(j[0])

        outputs = outputs.view(outputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        loss = criterion(outputs, targets)

        predict = outputs > 0.5
        predict_acc = [torch.equal(a, b) for a, b in zip(predict, targets)]
        predict_sens = [torch.equal(a, b) for a, b in zip(predict, targets) if torch.equal(b, torch.FloatTensor([1., 0.]).cuda())]
        predict_spec = [torch.equal(a, b) for a, b in zip(predict, targets) if torch.equal(b, torch.FloatTensor([0., 1.]).cuda())]
        predict_prec = [torch.equal(a, b) for a, b in zip(predict, targets) if torch.equal(a, torch.FloatTensor([1., 0.]).cuda())]
        # print(len(predict_sens), len(predict_spec), len(predict_prec))

        losses.update(loss.item(), inputs.size(0))
        acc.update(sum(predict_acc) / len(predict_acc), len(predict_acc))
        if len(predict_sens) != 0:
            sens.update(sum(predict_sens) / len(predict_sens), len(predict_sens))
        if len(predict_spec) != 0:
            spec.update(sum(predict_spec) / len(predict_spec), len(predict_spec))
        if len(predict_prec) != 0:
            prec.update(sum(predict_prec) / len(predict_prec), len(predict_prec))
        
        progress_bar(batch_idx, len(validloader), 'Loss: %.2f | Acc: %.2f' % (losses.avg, acc.avg))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
    # 抽样1000次计算CI
    labelList_CI = []
    predList_CI = []
    auc_values = []
    idx_list = list(np.arange(len(labelList)))
    for i in np.arange(1000):
        idx = random.sample(idx_list, int(len(labelList)*0.7))
        idx = list(idx)
        for j in idx:
            labelList_CI.append(labelList[j])
            predList_CI.append(predList[j])
        labelArray = np.array(labelList_CI)
        predArray = np.array(predList_CI)
        roc_auc = sklearn.metrics.roc_auc_score(labelArray, predArray)
        auc_values.append(roc_auc)
    ci_95 = np.percentile(auc_values, (2.5, 97.5))

    # 计算FPR、TPR, 输出AUC(95%CI)
    fpr, tpr, _ = sklearn.metrics.roc_curve(np.array(labelList), np.array(predList))
    auc = round(sklearn.metrics.auc(fpr, tpr), 4)
    ci_95 = (round(ci_95[0], 4), round(ci_95[1], 4))

    return (losses.avg, acc.avg, sens.avg, spec.avg, prec.avg, auc, ci_95)
    # return (losses.avg, acc.avg)


def adjust_learning_rate(optimizer, epoch):
    global common_config
    if epoch in common_config['schedule']:
        common_config['lr'] *= common_config['gamma']
        for param_group in optimizer.param_groups:
            param_group['lr'] = common_config['lr']


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Classify for Medical Image')
    # model related, including  Architecture, path, datasets
    parser.add_argument('--config-file', type=str, default='configs/config_classify.yaml')
    parser.add_argument('--gpu-id', type=str, default='1,2')
    parser.add_argument('--visualize', action='store_false')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main(args.config_file)