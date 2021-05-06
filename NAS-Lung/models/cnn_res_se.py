import sys
import os
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.se_layers import *
from models.net_sphere import *
def conv_block_3d(*args, **kwargs) -> nn.Module:
    return nn.Sequential(nn.Conv3d(*args, **kwargs),
                nn.BatchNorm3d(args[1] if len(args)>1 else kwargs["out_channels"]))

def conv_block_3d_relu(*args, **kwargs):
    return nn.Sequential(conv_block_3d(*args, **kwargs), nn.ReLU())
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = conv_block_3d_relu(in_channels, out_channels, kernel_size = kernel_size, padding = padding)
        self.conv2 = conv_block_3d(out_channels, out_channels, kernel_size = kernel_size, padding= kernel_size // 2)
        self.conv3 = conv_block_3d(in_channels,  out_channels, kernel_size = 1, padding= 0)
        self.relu  = nn.ReLU()
    
    def forward(self, inputs):
        residual = inputs
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = out + self.conv3(residual)
        out = self.relu(out)
        return out

class SEResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super(SEResBlock, self).__init__()
        self.residual_block = ResidualBlock(in_channels, out_channels, kernel_size)
        self.se_layer = SELayer3D(out_channels)
    
    def forward(self, inputs):
        out = self.residual_block(inputs)
        out = self.se_layer(out)
        return out

class ConvResSE(nn.Module):
    def __init__(self, config, softmax = "angle"):
        super(ConvResSE, self).__init__()
        self.config= config
        self.conv1 = SEResBlock(1, 4, kernel_size = 3)
        self.conv2 = SEResBlock(4, 4, kernel_size = 3)
        layers = []
        last_channel = 4
        pool_shape   = 2
        for i, stage in enumerate(config["stage"]):
            for n_channels in stage:
                layers.append(SEResBlock(last_channel, n_channels))
                last_channel = n_channels
        self.layers     = nn.Sequential(*layers)
        self.avg_pooling=nn.AdaptiveAvgPool3d(pool_shape)
        self.fc = AngleLinear(in_features=pool_shape ** 3 * config["stage"][-1][-1], out_features=2)
        # self.fc         =nn.Linear(pool_shape ** 3 * config["stage"][-1][-1], 2)
        
    def forward(self, inputs):
        batch_size,_, D, W, H = inputs.shape
        v1 = torch.swapaxes(inputs, 2, 3)
        v2 = torch.swapaxes(inputs, 2, 4)
        v3 = torch.swapaxes(inputs, 3, 4)
        out_lst = []
        out_fc_lst = []
        spc_att_lst= []
        for v in [inputs, v1,v2,v3]:
            out = self.conv1(v)
            out = self.conv2(out)
            # out, sp_att = self.first_cbam(out,get_sp_attention = True)
            out = self.layers(out)
            out = self.avg_pooling(out)
            out = out.view(out.size(0), -1)
            out_fc = self.fc(out)
            out_lst.append(out)
            out_fc_lst.append(out_fc[0])
            # spc_att_lst.append(sp_att)
        out_fc = self.fc(out_lst[0])
        return (out_fc[0], out_fc, out_lst, out_fc_lst, spc_att_lst)

        # batch_size, C, D, W, H = inputs.shape
        # # out = self.slcs_attention(inputs)
        # out = self.conv1(out)
        # out = self.conv2(out)
        # out = self.layers(out)
        # out = self.avg_pool(out)
        # out = self.fc(out.view(batch_size, -1))
        # return out

class SlicesAttentionBlock(nn.Module):
    def __init__(self, n_slices = 32, slice_kernel_size = 3, fc_reduction = 2):
        super(SlicesAttentionBlock, self).__init__()
        self.slice_kernel_size = slice_kernel_size
        self.conv_slices= nn.Sequential(
            nn.Conv3d(1, 1, kernel_size = (1, slice_kernel_size, slice_kernel_size)),
            nn.BatchNorm3d(1),
            nn.ReLU()
        ) 
        self.global_pool = nn.AdaptiveAvgPool3d((n_slices, 1, 1))
        self.fc1 = nn.Linear(n_slices, n_slices//fc_reduction)
        self.fc2 = nn.Linear(n_slices//fc_reduction,n_slices)
        self.relu= nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs):
        batch_size, _, D, W, H = inputs.shape
        feat_2d = self.conv_slices(inputs)
        out     = self.global_pool(feat_2d) 
        slcs_weight = self.fc1(out.view(batch_size, -1))
        slcs_weight = self.relu(slcs_weight)
        slcs_weight = self.fc2(slcs_weight)
        slcs_weight = self.sigmoid(slcs_weight)
        slcs_weight = slcs_weight.view(batch_size,1,D,1,1)
        out = inputs * slcs_weight
        return out

if __name__ == "__main__":
    inputs = torch.randn(5,1,32,32,32)
    net = ConvResSE({"stage": [[4,4], [4,8]]})
    net(inputs)

    # slc_attention = SlicesAttentionBlock()
    # slc_attention(inputs)