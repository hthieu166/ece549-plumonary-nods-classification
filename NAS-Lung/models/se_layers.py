import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb

class SELayer3D(nn.Module):
    def __init__(self, num_chanels, r = 2):
        super(SELayer3D, self).__init__()
        self.r = r
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_chanels // r
        self.fc1 = nn.Linear(num_chanels, num_channels_reduced)
        self.fc2 = nn.Linear(num_channels_reduced, num_chanels)
        self.relu= nn.ReLU()
        self.sigmoid =nn.Sigmoid()
    
    def forward(self, inputs):
        batch_size, C, D, H, W = inputs.shape
        squeeze = self.avg_pool(inputs).view(batch_size, C)
        out     = self.fc1(squeeze)
        out     = self.relu(out)
        out     = self.fc2(out)
        out     = self.sigmoid(out)
        out     = inputs * out.view(batch_size, C, 1, 1, 1)
        return out
        
if __name__ == "__main__":
    inputs  =  Variable(torch.randn(3,10,32,32,32))
    se_layer= SELayer3D(10)
    se_layer(inputs) 