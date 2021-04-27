import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
from models.cnn_res import *
import ipdb

class ConvResMultiViewsV2(nn.Module):
    def __init__(self, config, softmax = "angle"):
        super(ConvResMultiViewsV2, self).__init__()
        print(config)
        self.conv1 = conv3d_same_size(in_channels=1, out_channels=4, kernel_size=3)
        self.conv2 = conv3d_same_size(in_channels=4, out_channels=4, kernel_size=3)
        self.config = config
        self.last_channel = 4
        self.first_cbam = ResCBAMLayer(4, 32)
        layers = []
        i = 0
        for stage in config:
            i = i+1
            stage_mods = []
            stage_mods.append(conv3d_pooling(self.last_channel, kernel_size=3, stride=2))
            for channel in stage:
                stage_mods.append(ResidualBlock(self.last_channel, channel))
                self.last_channel = channel
            stage_mods.append(ResCBAMLayer(self.last_channel, 32//(2**i)))
            layers.append(nn.Sequential(*stage_mods))
        self.layers = nn.ModuleList(layers)
        self.avg_pooling       = nn.AvgPool3d(kernel_size=4, stride=4)
        self.low_feat_conv     = conv3d_pooling(8, kernel_size=5, stride=2)
        self.combine_conv      = conv3d_same_size(16,8,3)
        # self.fc = nn.Linear(in_features=self.last_channel, out_features=2)
        if softmax == "angle":
            self.fc = AngleLinear(in_features=self.last_channel, out_features=2)
        elif softmax == "normal":
            self.fc = AngleLinear(in_features=self.last_channel, out_features=2, get_phi_theta = False)
        else:
            print("Softmax option does not support!")
            raise
    
    def forward(self, inputs):
        batch_size,_, D, W, H = inputs.shape
        out_lst = []
        #Pre-modules
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.first_cbam(out)
        o0 = self.layers[0](out)
        o1 = torch.swapaxes(o0, 2, 3)
        o2 = torch.swapaxes(o0, 2, 4)
        o3 = torch.swapaxes(o0, 3, 4)
        for o in [o0, o1, o2, o3]:
            for i, layer in enumerate(self.layers[1:]):
                o = layer(o)           
            o = self.avg_pooling(o)
            o = o.view(o.size(0), -1)
            out_lst.append(o)
        out_fc = self.fc(out_lst[0])
        return (out_fc[0], out_fc, out_lst)
if __name__ == "__main__":
    net = ConvResMultiViewsV2( [[4,4], [4,8], [8,8]])
    inputs = torch.randn((5, 1, 32, 32, 32))
    output = net(inputs)
    print(output)
