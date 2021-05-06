import numpy as np
from matplotlib import pyplot as plt
import os
import os.path as osp
import pandas as pd
import matplotlib.patches as patches
import seaborn as sns
def read_test_data(exp_id, fold):
    feats = np.load("../log/infer-%s/deep-feat-%d.npy" %(exp_id, fold))
    preds_all_views = np.load("../log/infer-%s/preds-%d.npy"%(exp_id, fold))
    
    sp_attn_maps  = np.load("../log/infer-%s/sp-att-%d.npy"%(exp_id, fold)).squeeze(2)
    preds = np.argmax(preds_all_views[0], axis=1)
    teidlst = []
    gts = []
    nod_ids = []

    with open(osp.join("../subsets", "subset{}.txt".format(str(fold)))) as fo:
        teidlst += [i.strip() for i in fo.readlines()]
    df = pd.read_csv('../data/annotationdetclsconvfnl_v3.csv',
                    names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])
    for sid, gt in zip(df['seriesuid'].tolist()[1:], df['malignant'].tolist()[1:]):
        if sid.split('-')[0] in teidlst:
            gts.append(int(gt))
            nod_ids.append(sid)
    gts = np.array(gts)
    nod_ids = np.array(nod_ids)
    
    return nod_ids, preds, gts, feats, sp_attn_maps

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
            
def plot_nodule_img(ax, idx,  nods_lst, gts, img_dir = "/home/hthieu/data/crop/"):
    nod_npy   = np.load(osp.join(img_dir, nods_lst[idx]+".npy"))
    ax.add_patch(patches.Rectangle((0,0),31, 31,linewidth=5, edgecolor="red" if gts[idx] == 1 else "green", facecolor='none'))
    ax.imshow(nod_npy[15,:,:], cmap = "bone")
    ax.axis("off")
    ax.set_title(str(idx))

def plot_feature_vector(ax, idx, feats,preds, gts):
    sns.heatmap(feats[:,idx,:].T, ax = ax,
                    vmin =0, vmax=2.5, cbar=False)
    ax.add_patch(patches.Rectangle((0,0),4, 8,linewidth=8, edgecolor="red" if gts[idx] == 1 else "green", facecolor='none'))
    ax.set_title("%d-[%d|%d]" %(idx, preds[idx], gts[idx]))
    ax.set_xticklabels(["o", "x", "y","z"])
    ax.vlines(np.arange(5), *ax.get_ylim(), color="yellow")
    
def plot_imgs_grid(viz_func, viz_lst, *kwarg, nrows=2, ncols=8):
    plt.figure()
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(ncols *4,nrows *4))
    for i in range(nrows):
        for j in range(ncols):
            idx = (i * ncols +j)
            if idx >= len(viz_lst):
                return
            viz_func(axs[i,j],  viz_lst[idx], *kwarg)

def get_view(nod_npy, view = 0):
    if view == 0:
        return nod_npy
    elif view == 1:
        return np.swapaxes(nod_npy,0,1)
    elif view == 2:
        return np.swapaxes(nod_npy,0,2)
    elif view == 3:
        return np.swapaxes(nod_npy,1,2)
    
def plot_nodule_sp_attention(ax, idx,  nods_lst, gts, sp_attn, view, img_dir = "/home/hthieu/data/crop/"):
    nod_npy   = np.load(osp.join(img_dir, nods_lst[idx]+".npy"))
    ax.add_patch(patches.Rectangle((0,0),31, 31,linewidth=5, edgecolor="red" if gts[idx] == 1 else "green", facecolor='none'))
    nod_npy_vis = get_view(nod_npy, view)
    ax.imshow(nod_npy_vis[15,:,:], cmap = "bone")
    ax.imshow(sp_attn[view, idx,:][15,:,:], alpha=0.4, cmap='rainbow')
    ax.axis("off")
    ax.set_title(str(idx))