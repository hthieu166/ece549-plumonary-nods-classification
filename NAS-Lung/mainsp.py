from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import transforms as transforms
from dataloader import lunanod
import os
import argparse
import time
from models.cnn_res import *
from models.dpn3d import *
from models.net_sphere import *
from models.cnn_res_se import *
from models.cnn_res_multi_view import *
from models.cnn_res_multi_view_v2 import *
from models.cnn_res_multi_view_v3 import *
from losses.loss_multiview import *
# from utils import progress_bar
from torch.autograd import Variable
import numpy as np
import ast
import pandas as pd
import glob
import os.path as osp
import ipdb
import tqdm
#Modified by hthieu
from code.train_utils import TrainUtils
from code.train_config import Config
print("Total cuda devices", torch.cuda.device_count())
# preprocesspath  = '/media/DATA/LUNA16/crop/'
preprocesspath  = '../../data/crop/'
dataframe       = pd.read_csv('./data/annotationdetclsconvfnl_v3.csv',
                        names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])
SUBSETS_DIR      = './subsets/'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--savemodel', type=str, default='', help='resume from checkpoint model')
parser.add_argument("--gpuids", type=str, default='all', help='use which gpu')
parser.add_argument('--num_workers', type=int, default=8)

parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam

parser.add_argument('--fold', type=int, default=-1, help="fold")
parser.add_argument('--mode', type=str, default="train", help = "training or testing mode")
parser.add_argument("--config_file", type=str, default=None, help = "path to config file")
parser.add_argument("--eval_mode", type=str, default= "1fold", help = "select mode for eval, 1fold or 5 folds")
parser.add_argument("--log_dir", type=str, default="./log", help = "select dir for saving logs/checkpoints")

args = parser.parse_args()

CROPSIZE = 32
gbtdepth = 1

blklst = []
use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
best_acc_gbt = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
pixvlu, npix = 0, 0

#### LOAD CONFIG ###
cfg = Config(args.config_file)
if args.mode == "train":
    log_dir = osp.join(args.log_dir, 
            cfg.experiment_id+"-fold-"+str(args.fold)) if args.eval_mode == "1fold" else osp.join(args.log_dir, cfg.experiment_id)
else:
    log_dir = osp.join(args.log_dir, 
            args.mode + cfg.experiment_id+"-fold-"+str(args.fold)) if args.eval_mode == "1fold" else osp.join(args.log_dir, cfg.experiment_id)
tu = TrainUtils(log_dir, ckpt_every = cfg.train_params["ckpt_every"], train_config_dict = cfg._config) 

#### FOLDS FOR EVALUATE ###
if args.eval_mode == "1fold":
    tu.log("Using 1 fold evaluation")
    fold = [args.fold]
else:
    tu.log("Using 5 folds evaluation")
    fold = [0,1,2,3,4]
    if args.fold != -1:
        print("Flag --fold should not be used in the 5 folds evaluation mode")

all_nods_npy = glob.glob(osp.join(preprocesspath,"*.npy"))
print("Total nodules ", len(all_nods_npy))

pixmean, pixstd = cfg.dataset_params["normalize"]["mean"], cfg.dataset_params["normalize"]["std"]

print('mean ' + str(pixmean) + ' std ' + str(pixstd))
# Datatransforms
print('==> Preparing data..')  # Random Crop, Zero out, x z flip, scale,
transform_train = transforms.Compose([
    # transforms.RandomScale(range(28, 38)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomYFlip(),
    transforms.RandomZFlip(),
    transforms.ZeroOut(4),
    transforms.ToTensor(),
    transforms.Normalize((pixmean), (pixstd)),  # need to cal mean and std, revise norm func
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((pixmean), (pixstd)),
])

# load data list
trfnamelst = []
trlabellst = []
trfeatlst = []
tefnamelst = []
telabellst = []
tefeatlst = []

alllst = dataframe['seriesuid'].tolist()[1:]
labellst = dataframe['malignant'].tolist()[1:]
crdxlst = dataframe['coordX'].tolist()[1:]
crdylst = dataframe['coordY'].tolist()[1:]
crdzlst = dataframe['coordZ'].tolist()[1:]
dimlst = dataframe['diameter_mm'].tolist()[1:]
# test id
teidlst = []
for test_fold in fold:
    with open(osp.join(SUBSETS_DIR, "subset{}.txt".format(str(test_fold)))) as fo:
        teidlst += [i.strip() for i in fo.readlines()]
print("Total test nodules: ",len(teidlst))
mxx = mxy = mxz = mxd = 0
for srsid, label, x, y, z, d in zip(alllst, labellst, crdxlst, crdylst, crdzlst, dimlst):
    feat = np.array([d]).astype(np.float32)
    if srsid.split('-')[0] in teidlst:
        tefnamelst.append(srsid + '.npy')
        telabellst.append(int(label))
        tefeatlst.append(feat)
    else:
        trfnamelst.append(srsid + '.npy')
        trlabellst.append(int(label))
        trfeatlst.append(feat)

trainset = lunanod(preprocesspath, trfnamelst, trlabellst, trfeatlst, train=True, download=True,
                   transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.train_params["batch_size"], shuffle=True, num_workers=args.num_workers)

testset = lunanod(preprocesspath, tefnamelst, telabellst, tefeatlst, train=False, download=True,
                  transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.train_params["batch_size"], shuffle=False, num_workers=args.num_workers)

# Model
tu.log('==> Building model..')
tu.log('args.savemodel : ' + args.savemodel)
if  cfg.model_name == "NAS":
    tu.log("Running NAS")
    if cfg.loss_name == "CrossEntropy":
        net = ConvRes(cfg.model_config["config"], softmax="normal")
    else:
        net = ConvRes(cfg.model_config["config"], softmax="angle")
elif cfg.model_name == "DPN3D":
    net = DPN92_3D()
elif cfg.model_name == "SE-RES":
    net = ConvResSE(cfg.model_config["config"])
elif cfg.model_name == "RES-MULTIVIEWS":
    net = ConvResMultiViews(cfg.model_config["config"])
elif cfg.model_name == "RES-MULTIVIEWS-V2":
    net = ConvResMultiViewsV2(cfg.model_config["config"])
elif cfg.model_name == "RES-MULTIVIEWS-V3":
    net = ConvResMultiViewsV3(cfg.model_config["config"])
else: 
    print("Unsupported model ", cfg.model_name, "!")
    raise  
tu.log("Running model: " + cfg.model_name)
if args.savemodel != "":
    print('==> Resuming from checkpoint..')
    checkpoint = tu.resume_from_ckpt(args.savemodel)
    best_acc = checkpoint['eval']['acc']
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint["net"].state_dict())
    print("net_loaded")

lr = cfg.train_params["init_lr"]

def get_lr(epoch):
    global lr
    if (epoch in cfg.train_params["decay_epochs"]):
        lr = lr * cfg.train_params["lr_decay"]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Decay learning rate to lr: {}.'.format(lr))

if use_cuda:
    net.cuda()
    if args.gpuids == 'all':
        device_ids = range(torch.cuda.device_count())
    else:
        device_ids = map(int, list(filter(str.isdigit, args.gpuids)))
    if (torch.cuda.device_count() == 1):
        device_ids = [torch.device('cuda:0')]
    else:
        device_ids = [torch.device('cuda:0'), torch.device('cuda:1')]
    print('gpu use' + str(device_ids))
    net = torch.nn.DataParallel(net, device_ids=device_ids)
    cudnn.benchmark = False  # True

if (cfg.loss_name == None):   
    take_first = False
    criterion = nn.CrossEntropyLoss()
elif cfg.loss_name == "CrossEntropy": 
    take_first = False
    criterion = nn.CrossEntropyLoss()
elif (cfg.loss_name == "AngleLoss"):
    take_first = True
    criterion = AngleLoss()
elif (cfg.loss_name == "MultiViews"):
    take_first = True
    criterion = MultiViewsLoss(cfg.loss)
elif (cfg.loss_name == "MultiViewsContrast"):
    take_first = True
    criterion = MultiViewsContrastLoss()
else:
    print("Loss function does not support!")
    raise 

tu.log("Using loss: " + cfg.loss_name)

optimizer = optim.Adam(net.parameters(), lr=cfg.train_params["init_lr"], betas=(args.beta1, args.beta2))

# Training
def train(epoch):
    print("\nEpoch: " + str(epoch))
    net.train()
    get_lr(epoch)
    train_loss = 0
    correct = 0
    total = 0
    with tqdm.tqdm(total=len(trainloader)) as pbar:
        for batch_idx, (inputs, targets, feat) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets) 
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
            if take_first == True:
                outputs = outputs[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            pbar.update()

    train_acc = correct.data.item() / float(total)
    tu.add_train_info(epoch, {
        "acc": train_acc, "lr": lr, "loss": train_loss})

def test(epoch, infer = False):
    epoch_start_time = time.time()
    global best_acc
    global best_acc_gbt
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    TP = FP = FN = TN = 0
    test_feats = []
    test_fcs   = []
    test_sp_att= []
    for batch_idx, (inputs, targets, feat) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs, requires_grad=False), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # outputs = net(inputs, multi_view_feats = True)
        # loss = criterion(outputs[1], targets)
        test_loss += loss.data.item()
        if infer == True:
            test_feats.append(torch.stack(outputs[2]).cpu().detach().numpy())
            test_fcs.append  (torch.stack(outputs[3]).cpu().detach().numpy())
            test_sp_att.append(torch.stack(outputs[4]).cpu().detach().numpy())
        if take_first == True:
            outputs = outputs[0]
        
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        TP += ((predicted == 1) & (targets.data == 1)).cpu().sum()
        TN += ((predicted == 0) & (targets.data == 0)).cpu().sum()
        FN += ((predicted == 0) & (targets.data == 1)).cpu().sum()
        FP += ((predicted == 1) & (targets.data == 0)).cpu().sum()
    if infer == True:
        test_feats = np.hstack(test_feats)
        test_fcs   = np.hstack(test_fcs)
        test_sp_att= np.hstack(test_sp_att)
        out_dir = osp.join("log", "infer-"+cfg.experiment_id)
        os.makedirs(out_dir, exist_ok = True)
        np.save(osp.join(out_dir, "deep-feat-%s" % str(args.fold)), test_feats)
        np.save(osp.join(out_dir, "preds-%s" % str(args.fold)), test_fcs)
        np.save(osp.join(out_dir, "sp-att-%s" % str(args.fold)), test_sp_att)
        
    # Save checkpoint.
    acc = 100. * correct.data.item() / total
    tpr = 100. * TP.data.item() / (TP.data.item() + FN.data.item())
    fpr = 100. * FP.data.item() / (FP.data.item() + TN.data.item())
    state = {
        'net': net.module if use_cuda else net,
        'epoch': epoch,
        'eval':{
            'acc': acc,
            'tpr': tpr,
            'fpr': fpr
        }
    }
    tu.add_test_ckpt(epoch, state)

if __name__ == '__main__':
    if args.mode == "train":
        for epoch in range(start_epoch + 1, start_epoch + cfg.train_params["n_epochs"] + 1):
            train(epoch)
            test(epoch)
    elif args.mode == "test":
        test(0)
    elif args.mode == "infer":
        test(0, infer=True)


    #CALCULATE MEAN & STD
# for fname in all_nods_npy:
#     if fname.endswith('.npy'):
#         if fname[:-4] in blklst: continue
#         data = np.load(fname)
#         pixvlu += np.sum(data)
#         # print("data.shape = " + str(data.shape))
#         npix += np.prod(data.shape)
# pixmean = pixvlu / float(npix)
# pixvlu = 0
# for fname in all_nods_npy:
#     if fname.endswith('.npy'):
#         if fname[:-4] in blklst: continue
#         data = np.load(fname) - pixmean
#         pixvlu += np.sum(data * data)

# pixstd = np.sqrt(pixvlu / float(npix))
# # pixstd /= 255
# print(pixmean, pixstd)

   # tu.add_new_checkpoint(state, epoch)

    # if acc > best_acc:
    #     logging.info('Saving..')
    #     state = {
    #         'net': net.module if use_cuda else net,
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir(savemodelpath):
    #         os.mkdir(savemodelpath)
    #     torch.save(state, savemodelpath + 'ckpt.t7')
    #     best_acc = acc
    # logging.info('Saving..')
    # state = {
    #     'net': net.module if use_cuda else net,
    #     'acc': acc,
    #     'epoch': epoch,
    # }
    # if not os.path.isdir(savemodelpath):
    #     os.mkdir(savemodelpath)
    # #Save checkpoint after a fixed period
    # tu.add_new_checkpoint(state, epoch)
    # if epoch % 50 == 0:
    #     torch.save(state, savemodelpath + 'ckpt' + str(epoch) + '.t7')
    # best_acc = acc
   

    # print('teacc ' + str(acc) + ' bestacc ' + str(best_acc))
    # print('tpr ' + str(tpr) + ' fpr ' + str(fpr))
    # print('Time Taken: %d sec' % (time.time() - epoch_start_time))
    # logging.info(
    #     'teacc ' + str(acc) + ' bestacc ' + str(best_acc))
    # logging.info(
    #     'tpr ' + str(tpr) + ' fpr ' + str(fpr))

