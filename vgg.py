from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models

class BatchNormVgg19(torch.nn.Module):
    def __init__(self):
        super(BatchNormVgg19, self).__init__()
        self.layers = models.vgg19_bn(pretrained=True).features
        self.slice1 = nn.Sequential(*list(self.layers.children())[0:6]) 
        self.slice2 = nn.Sequential(*list(self.layers.children())[6:13]) 
        self.slice3 = nn.Sequential(*list(self.layers.children())[13:20]) 
        self.slice4 = nn.Sequential(*list(self.layers.children())[20:33]) 
        self.slice5 = nn.Sequential(*list(self.layers.children())[33:46]) 
        for param in self.parameters():
                param.requires_grad = False
    def forward(self, x):
        out_1_2 = self.slice1(x)
        out_2_2 = self.slice2(out_1_2)
        out_3_2 = self.slice3(out_2_2)
        out_4_2 = self.slice4(out_3_2)
        out_5_2 = self.slice5(out_4_2)
        return [out_1_2, out_2_2, out_3_2, out_4_2, out_5_2]

class Vgg19(torch.nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        self.layers = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential(*list(self.layers.children())[0:4]) 
        self.slice2 = nn.Sequential(*list(self.layers.children())[4:9]) 
        self.slice3 = nn.Sequential(*list(self.layers.children())[9:14]) 
        self.slice4 = nn.Sequential(*list(self.layers.children())[14:23]) 
        self.slice5 = nn.Sequential(*list(self.layers.children())[23:32]) 
        for param in self.parameters():
                param.requires_grad = False
    def forward(self, x):
        out_1_2 = self.slice1(x)
        out_2_2 = self.slice2(out_1_2)
        out_3_2 = self.slice3(out_2_2)
        out_4_2 = self.slice4(out_3_2)
        out_5_2 = self.slice5(out_4_2)
        return [out_1_2, out_2_2, out_3_2, out_4_2, out_5_2]

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16_bn(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out