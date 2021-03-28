import numpy as np
import glob
import os.path as osp
import argparse
import torch
import ipdb
import pandas as pd
import os 
def folds_evaluate(args):
    folds = sorted(glob.glob(osp.join(args.log_dir, "{}-*".format(args.exp_name))))
    assert len(folds) > 0, "No folds validation found!"
    eval_ = {}
    for fold in folds:
        model = osp.join(fold, "best.model")
        name  = osp.basename(fold)
        ckpt  = torch.load(model)
        eval_[name] = ckpt["best_acc"]
    best_acc =  [eval_[k] for k in eval_.keys()]
    out_df = pd.DataFrame({
        "fold":list(eval_.keys()), "best_acc": best_acc})
    out_df.set_index("fold")
    best_acc = np.array(best_acc)
    out_df.loc['mean'] = out_df.mean()
    print("Mean Acc. ", best_acc.mean())
    os.makedirs(args.out_dir, exist_ok = True)
    out_df.to_csv(osp.join(args.out_dir, "{}.csv".format(args.exp_name)))

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='nas-base-fold', help='name of the experiment')
    parser.add_argument('--log_dir', type=str, default='/home/hthieu/plumonary_nods_classification/ece549-plumonary-nods-classification/NAS-Lung/log', 
                        help='path to the log directory')
    parser.add_argument("--out_dir", type=str, default="../eval_result", help='path ')
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    folds_evaluate(args)