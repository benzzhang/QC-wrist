'''
Date: 2023-04-21 10:52:12
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-07-21 14:28:56
FilePath: /QC-wrist/utils/misc.py
Description: 
'''
import errno
import os
import torch
import shutil

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def save_checkpoint(state, is_best, save_path, ckp_name='checkpoint.pth.tar', best_name='model_best.pth.tar'):
    filepath = os.path.join(save_path, ckp_name)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_path, best_name))

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
