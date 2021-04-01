import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
from models.net_sphere import *

class MultiViewsLoss(nn.Module):
    def __init__(self, alpha = .5):
        super(MultiViewsLoss, self).__init__()
        self.alpha = alpha
        self.sft_max_angle_loss = AngleLoss()
        self.mse = nn.MSELoss()
    
    def forward(self, outputs, targets):
        lss1 = self.sft_max_angle_loss(outputs[1], targets)
        embed_vects = outputs[2]
        lss2 = 0
        for i in range(1, len(embed_vects)):
            lss2 += self.mse(embed_vects[0], embed_vects[i])
        lss2 /= (len(embed_vects)-1)
        return self.alpha * lss1 + (1-self.alpha) * lss2