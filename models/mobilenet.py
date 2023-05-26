'''
Date: 2023-04-23 13:43:23
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-05-23 17:36:18
FilePath: /QC-wrist/models/mobilenet.py
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

__all__ = ['MobileNet_v2_mulcls', 'MobileNet_v2_onecls']

class MobileNet_v2_mulcls(nn.Module):
    def __init__(self, n_foreign_classes=2, n_position_classes=3):
        super().__init__()
        self.base_model = models.mobilenet_v2().features  # take the model without classifier
        last_channel = models.mobilenet_v2().last_channel  # size of the layer before classifier

        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for our outputs
        self.foreign = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=n_foreign_classes),
            nn.Softmax(dim=1)
        )
        self.position = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=n_position_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)

        return {
            'foreign': self.foreign(x),
            'position': self.position(x)
        }
    
'''
    用于二分类任务的mobilenet
'''
class MobileNet_v2_onecls(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.base_model = models.mobilenet_v2().features  # take the model without classifier
        last_channel = models.mobilenet_v2().last_channel  # size of the layer before classifier

        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for our outputs
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            # nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            # nn.ReLU(),
            nn.Linear(in_features=32, out_features=num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x