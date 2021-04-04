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
    def __init__(self, config = None):
        super(MultiViewsLoss, self).__init__()
        self.config = config
        alpha = .5
        cls_loss_name = "AngleLoss"
        if (self.config != None):
            if "cls_loss" in self.config["params"]:  
                cls_loss_name = self.config["params"]["cls_loss"]
            if "alpha" in self.config["params"]:
                alpha = self.config["params"]["alpha"]
        
        self.alpha  = alpha
        self.cls_loss_name = cls_loss_name
        if self.cls_loss_name == "AngleLoss":
            self.cls_loss = AngleLoss()
        elif self.cls_loss_name == "CrossEntropy":
            self.cls_loss = nn.CrossEntropyLoss()
        else:
            print("Classification loss in params does not support")
            raise
        
        self.mse = nn.MSELoss()
    
    def forward(self, outputs, targets):
        if self.cls_loss_name == "AngleLoss":
            lss1 = self.cls_loss(outputs[1], targets)
        elif self.cls_loss_name == "CrossEntropy":
            lss1 = self.cls_loss(outputs[1][0], targets)
            
        embed_vects = outputs[2]
        lss2 = 0
        for i in range(1, len(embed_vects)):
            lss2 += self.mse(embed_vects[0], embed_vects[i])
        lss2 /= (len(embed_vects)-1)
        return self.alpha * lss1 + (1-self.alpha) * lss2

class MultiViewsPredLoss(nn.Module):
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

class MultiViewsContrastLoss(nn.Module):
    def __init__(self, weights = [1,1,1]):
        super(MultiViewsContrastLoss, self).__init__()
        self.weights = weights
        self.sft_max_angle_loss = AngleLoss()
        self.mse = nn.MSELoss()
    def forward(self, outputs, targets):
        #Term 1
        lss1 = self.sft_max_angle_loss(outputs[1], targets)
        #Term 2
        embed_vects = outputs[2]
        lss2 = 0
        for i in range(1, len(embed_vects)):
            lss2 += self.mse(embed_vects[0], embed_vects[i])
        lss2 /= (len(embed_vects)-1)
        #Term 3
        represented_vects = torch.mean(torch.stack(embed_vects,dim = 1), dim = 1)
        cls0_vect = represented_vects[targets == 0].mean(dim=0)
        cls1_vect = represented_vects[targets == 0].mean(dim=0)
        lss3 = self.mse(cls0_vect, cls1_vect)
        #Final loss
        fnlss = (self.weights[0] * lss1 + self.weights[1] * lss2 - self.weights[2] * lss3)/sum(self.weights)
        return fnlss

if __name__ == "__main__":
    targets = torch.Tensor([1,0,0,1,0])
    batch_size = len(targets)
    outputs = ( torch.randn(batch_size, 2),
                (torch.randn(batch_size, 2),torch.randn(batch_size, 2)),
                [torch.randn(batch_size,8) for i in range(4)])
    
    lss = MultiviewsLossContrast()
    print(lss(outputs, targets))