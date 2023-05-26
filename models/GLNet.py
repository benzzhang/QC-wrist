'''
Author: Peng Bo
Date: 2022-04-27 16:09:11
LastEditTime: 2022-06-22 17:23:53
Description:

'''
import torch
import torch.nn as nn
from .unet_dw  import UNetDW
from .globalNet import globalNet

class GLNet(nn.Module):
    ''' global and local net '''
    def __init__(self, num_classes, local_net='unet_dw'):
        super(GLNet, self).__init__()
        if local_net == 'unet_dw':
            self.localNet = UNetDW(num_classes=num_classes)
        else:
            exit(0)
        self.globalNet = globalNet(3, num_classes)

    def forward(self, x):
        local_feature = self.localNet(x)
        fuse = torch.cat((x, local_feature), dim=1)
        global_feature = self.globalNet(fuse)
        return global_feature*local_feature


if __name__ == "__main__":
    net = GLNet(num_classes=2, local_net='att_unet')
    import numpy as np
    input = np.random.rand(1, 3, 512, 512).astype(np.float32)
    input = torch.FloatTensor(torch.from_numpy(input))
    input = torch.autograd.Variable(input)
    output = net(input)
    import pdb
    pdb.set_trace()