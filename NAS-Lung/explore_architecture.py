import torch as T
from models.cnn_res import *
import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
    ckpt = T.load("log/ckpt/model-1-ckpt-5.t7")
    # ckpt = T.load("log/nas-base-fold-0/0020.model", map_location="cpu")

    # print(ckpt)
    # net = ConvRes([[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]])
    # print(net)