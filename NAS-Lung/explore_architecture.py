import torch as T
from models.cnn_res import *
import os
from torch.autograd import Variable
import ipdb 
import numpy as np
if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
    all_acc = []
    all_ckpts=[]
    for i in range(1,6):
        # ckpt = T.load("log/ckpt/Model-1/checkpoint-{}/ckpt.t7".format(i))
        ckpt = T.load("log/nas-model-1-angle-expr-4-fold-{}/best.model".format(i))
        all_acc.append(ckpt["eval"]["best_acc"])
        all_ckpts.append(ckpt["epoch"])
    print(all_acc)
    print(all_ckpts)

    # all_acc = np.array(all_acc)
    # print("Mean: ",all_acc.mean())
    # print(all_acc)
    # ckpt = T.load("log/ckpt/Model-1/checkpoint-1/ckpt.t7")
    # print(ckpt)
    # ckpt = T.load("log/nas-model-1-cross-entropy-fold-0/best.model", map_location="cpu")
    # print(ckpt)

    # x   = Variable(torch.randn(1,1,32,32,32))
    # net = ConvRes([[4,4], [4,8], [8,32]])
    # y   = net(x)
    # print(y)
    # ipdb.set_trace()

    # net.load_state_dict(ckpt)
    # for i in net.parameters():
    #     print(i.shape)
    # for comp1, comp2 in zip(net.parameters(), ckpt["net"].parameters()):
        # print(comp1.shape == comp2.shape)
    # print()

    # 87.85046729 85.58558559 89.32038835 89.         92.39130435