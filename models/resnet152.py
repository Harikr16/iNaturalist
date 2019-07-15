# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pdb
import math

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

__all__ = [ 'ResNet', 'resnet']
 
# (number of filters, kernel size, stride, pad)
CFG = {
    '2012': [(96, 11, 4, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1), 'M']
}


class ResNet(nn.Module):
    def __init__(self, features, num_classes, sobel):
        super(ResNet, self).__init__()
        self.features = features
        # self.classifier = nn.Sequential(nn.Dropout(0.5),
        #                     nn.Linear(256 * 6 * 6, 4096),
        #                     nn.ReLU(inplace=True),
        #                     nn.Dropout(0.5),
        #                     nn.Linear(4096, 4096),
        #                     nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(nn.ReLU(inplace = True))

        self.top_layer = nn.Linear(2048, num_classes)
        # self._initialize_weights()

        if sobel:
            grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
            grayscale.weight.data.fill_(1.0 / 3.0)
            grayscale.bias.data.zero_()
            sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
            sobel_filter.weight.data[0, 0].copy_(
                torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            )
            sobel_filter.weight.data[1, 0].copy_(
                torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            )
            sobel_filter.bias.data.zero_()
            self.sobel = nn.Sequential(grayscale, sobel_filter)
            for p in self.sobel.parameters():
                p.requires_grad = False
        else:
            self.sobel = None

    def forward(self, x):
        # try:
        if self.sobel:
            x_ = self.sobel(x)
        else:
            x_ = x
        x_ = self.features(x_)
        # pdb.set_trace()
        x_ = self.pool(x_)
        x_ = x_.view(x_.size(0), -1)
        x_ = self.classifier(x_)
        if self.top_layer:
            x_ = self.top_layer(x_)
        return x_
        # except:
        #     pdb.set_trace()

    # def _initialize_weights(self):
    #     for y, m in enumerate(self.modules()):
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             for i in range(m.out_channels):
    #                 m.weight.data[i].normal_(0, math.sqrt(2. / n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             m.weight.data.normal_(0, 0.01)
    #             m.bias.data.zero_()


# def make_layers_features(cfg, input_dim, bn):
#     layers = []
#     in_channels = input_dim
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1], stride=v[2], padding=v[3])
#             if bn:
#                 layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v[0]
#     return nn.Sequential(*layers)


def resnet(sobel=False, bn=True, out=1000):
    dim = 2 + int(not sobel)
    # model = AlexNet(make_layers_features(CFG['2012'], dim, bn=bn), out, sobel)
    model = models.resnet152(pretrained=True)
    if sobel:
        model = ResNet(nn.Sequential(nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),*list(model.children())[1:-2]), num_classes =out, sobel = sobel)
    else:
        model = ResNet(nn.Sequential(*list(model.children())[:-2]), num_classes =out, sobel = sobel)
    # pdb.set_trace()
    return model
