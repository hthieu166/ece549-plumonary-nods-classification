import torch
import torch.nn as nn
import torch.nn.functional as F
from .se_layers import *
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
        self.conv1 = conv_block_3d_relu(1, 4, kernel_size = 3, padding = 1)
        self.conv2 = conv_block_3d_relu(4, 4, kernel_size = 3, padding = 1)
        layers = []
        last_channel = 4
        pool_shape   = 2
        for i, stage in enumerate(config["stage"]):
            for n_channels in stage:
                layers.append(SEResBlock(last_channel, n_channels))
                last_channel = n_channels
        self.layers = nn.Sequential(*layers)
        self.avg_pool=nn.AdaptiveAvgPool3d(pool_shape)
        self.fc      =nn.Linear(pool_shape ** 3 * config["stage"][-1][-1], 2)
        
    def forward(self, inputs):
        batch_size, C, D, W, H = inputs.shape
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.layers(out)
        out = self.avg_pool(out)
        out = self.fc(out.view(batch_size, -1))
        return out


if __name__ == "__main__":
    inputs = torch.randn(5,1,32,32,32)
    net = ConvResSE({"stage": [[4,4], [4,8]]})
    net(inputs)