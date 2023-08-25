'''
Date: 2023-04-23 13:47:40
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-08-08 15:53:58
FilePath: /QC-wrist/losses/loss.py
'''
import torch
from torch.nn import Module
from torch import nn
from torch.nn import functional as F

__all__ = ['mobilenet_mulcls_loss', 'MaskBCELoss']

class mobilenet_mulcls_loss(nn.Module):
    def __init__(self):
        super(mobilenet_mulcls_loss, self).__init__()

    def forward(self, net_output, ground_truth):
        foreign_loss = F.cross_entropy(net_output['foreign'], ground_truth['foreign_labels'])
        position_loss = F.cross_entropy(net_output['position'], ground_truth['foreign_labels'])
        loss = foreign_loss + position_loss
        return loss, {'foreign': foreign_loss, 'position': position_loss}

class MaskBCELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(MaskBCELoss, self).__init__()
        self.pos_weight = 1
        self.eps = 1e-6
        self.reduction = reduction

    def forward(self, logits, target, mask):
        # logits: [N, *], target: [N, *]
        loss = - self.pos_weight * (target * torch.log(logits+self.eps) + (1 - target) * torch.log(1 - logits - self.eps))
        each_loss = torch.sum(loss, dim=(2,3))

        if self.reduction == 'mean':
            loss = torch.mean(torch.mean(loss, dim=(2,3)) * mask)
        elif self.reduction == 'sum':
            loss = torch.sum(torch.sum(loss, dim=(2,3)) * mask)
        elif self.reduction == 'keep':
            loss = torch.sum(torch.sum(loss, dim=(2,3)) * mask, dim=0) / torch.sum(mask, dim=0)
            #loss = torch.sum(torch.sum(loss, dim=(2,3)) * mask, dim=0)

        return loss