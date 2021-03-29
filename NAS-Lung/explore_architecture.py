import torch as T
from models.cnn_res import *
import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
    # ckpt = T.load("log/ckpt/model-1-ckpt-5.t7")
    ckpt = T.load("log/nas-model-1-fold-1/best.model", map_location="cpu")
    print(ckpt)
    # net = ConvRes([[4,4], [4,8], [8,8]])
    # net.load_state_dict(ckpt)
    # for i in net.parameters():
    #     print(i.shape)
    # for comp1, comp2 in zip(net.parameters(), ckpt["net"].parameters()):
        # print(comp1.shape == comp2.shape)
    # print()