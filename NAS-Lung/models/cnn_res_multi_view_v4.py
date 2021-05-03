import typing
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
from models.cnn_res import *
class MultiViewFusionHead(nn.Module):
    def __init__(self):
        super(MultiViewFusionHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, inputs):
        print(inputs[0].shape)
        # print(torch.stack(inputs, dim=1).shape)
        ipdb.set_trace()

class ConvResMultiViewsV4(ConvRes):
    def __init__(self, config, softmax = "angle"):
        print(config)
        super(ConvResMultiViewsV4, self).__init__(config, softmax = "angle")
        self.view_attention_head = MultiViewFusionHead()
    
    def forward(self, inputs):
        batch_size,_, D, W, H = inputs.shape
        v1 = torch.swapaxes(inputs, 2, 3)
        v2 = torch.swapaxes(inputs, 2, 4)
        v3 = torch.swapaxes(inputs, 3, 4)
        out_lst = []
        out_fc_lst = []
        out_feat_map_lst=[]
        for v in [inputs, v1,v2,v3]:
            out = self.conv1(v)
            out = self.conv2(out)
            out = self.first_cbam(out)
            out = self.layers(out)
            out_feat_map_lst.append(out)
            out = self.avg_pooling(out)
            out = out.view(out.size(0), -1)
            out_fc = self.fc(out)
            out_lst.append(out)
            out_fc_lst.append(out_fc[0])
        
        combine_feat = self.view_attention_head(out_lst)
        out_fc = self.fc(combine_feat)
        return (out_fc[0], out_fc, out_lst, out_fc_lst)
if __name__ == "__main__":
    net = ConvResMultiViewsV4( [[4,4], [4,8], [8,8]])
    inputs = torch.randn((5, 1, 32, 32, 32))
    output = net(inputs)
    print(output)
