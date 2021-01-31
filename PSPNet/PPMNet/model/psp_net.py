#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 21:46:52 2021

@author: rajiv
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model.DRN import drn_d_105


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.LeakyReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(True)
        )

    def forward(self, x):
        return self.conv(x)
    
    
class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class Classifier(nn.Module):
    """Classifier"""
    
    def __init__(self, dw_channels, num_classes, stride = 1, **kwargs):
        super(Classifier, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv2d(dw_channels, num_classes, 1)
            )
    
    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x


class PSPNet(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(PSPNet, self).__init__()
        drn = drn_d_105()
        drn = list(drn.children())
        self.conv = nn.Sequential(*drn[:-2])
        self.ppm = PyramidPooling(512, 512)
        self.classifier = Classifier(512, num_classes)
    
    def forward(self, x):
        size = x.size()[2:]
        x = self.conv(x)
        ppm_out = self.ppm(x)
        x = ppm_out + x
        x = self.classifier(x)
        out = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        out.append(x)
        return tuple(out)
        
        

def get_model(pretrained=False, root="./weights", **kwargs):
    model = PSPNet(num_classes=19)
    return model
