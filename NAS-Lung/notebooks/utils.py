import numpy as np
from matplotlib import pyplot as plt
import os
import os.path as osp

def plot_slices(nod_npy, axis = 0):
    fig, axs = plt.subplots(nrows=4,ncols=4,figsize=(20,20))
    for i in range(4):
        for j in range(4):
            idx = (i * 4 +j)*2
            if axis == 0:
                im = nod_npy[idx,:,:]
            elif axis == 1:
                im = nod_npy[:,idx,:]
            else:
                im = nod_npy[:,:,idx]
            axs[i][j].imshow(im, cmap = "bone")
            axs[i][j].axis("off")
            axs[i][j].set_title(str(idx))