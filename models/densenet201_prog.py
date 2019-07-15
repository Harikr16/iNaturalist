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

__all__ = [ 'DenseNet', 'densenet']
 
# (number of filters, kernel size, stride, pad)
CFG = {
    '2012': [(96, 11, 4, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1), 'M']
}


class DenseNet(nn.Module):
    def __init__(self, features, num_classes, sobel):
        super(DenseNet, self).__init__()
        self.full_features = features
        # self.classifier = nn.Sequential(nn.Dropout(0.5),
        #                     nn.Linear(256 * 6 * 6, 4096),
        #                     nn.ReLU(inplace=True),
        #                      nn.Dropout(0.5),
        #                     nn.Linear(4096, 4096),
        #                     nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(nn.ReLU(inplace = True))
        self.top_layer = nn.Linear(1920, num_classes)
        self.blocks  = self.get_blocks()

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
        x_ = self.pool(x_)
        x_ = x_.view(x_.size(0), -1)
        x_ = self.classifier(x_)
        if self.top_layer:
            x_ = self.top_layer(x_)
        return x_

    def get_bloacks(self):
        idx = []
        for i in range(len(self.features)):
            for name,module in self.features[i].named_modules():
                if 'denselayer' in name:
                        idx.append(i)
                        break
        blocks = [self.features[:idx[0]+1]



def densenet(sobel=False, bn=True, out=1000):
    dim = 2 + int(not sobel)
    # model = AlexNet(make_layers_features(CFG['2012'], dim, bn=bn), out, sobel)
    model = models.densenet201(pretrained=True)
    # pdb.set_trace()
    if sobel:
        model = DenseNet(nn.Sequential(nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),*list(model.features[1:])), num_classes =out, sobel = sobel)
    else:
        model = DenseNet(nn.Sequential(*list(model.children())[:-1]), num_classes =out, sobel = sobel)
    # pdb.set_trace()
    return model

