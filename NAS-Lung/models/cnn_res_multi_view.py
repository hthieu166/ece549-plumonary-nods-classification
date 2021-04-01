import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
from models.cnn_res import *

class ConvResMultiViews(ConvRes):
    def __init__(self, config, softmax = "angle"):
        print(config)
        super(ConvResMultiViews, self).__init__(config, softmax = "angle")
    
    def forward(self, inputs):
        batch_size,_, D, W, H = inputs.shape
        v1 = torch.swapaxes(inputs, 2, 3)
        v2 = torch.swapaxes(inputs, 2, 4)
        v3 = torch.swapaxes(inputs, 3, 4)
        out_lst = []
        for v in [inputs, v1,v2,v3]:
            out = self.conv1(v)
            out = self.conv2(out)
            out = self.first_cbam(out)
            out = self.layers(out)
            out = self.avg_pooling(out)
            out = out.view(out.size(0), -1)
            out_lst.append(out)
        out_fc = self.fc(out_lst[0])
        return (out_fc[0], out_fc, out_lst)
if __name__ == "__main__":
    net = ConvResMultiViews( [[4,4], [4,8], [8,8]])
    inputs = torch.randn((5, 1, 32, 32, 32))
    output = net(inputs)
    print(output)
