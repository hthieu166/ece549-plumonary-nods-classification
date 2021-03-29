import numpy as np
import glob
import os.path as osp
import argparse
import torch
import ipdb
import pandas as pd
import os 
def folds_evaluate(args):
    group = osp.join(args.log_dir, "{}-*".format(args.exp_name))
    folds = sorted(glob.glob(group))
    assert len(folds) > 0, "No folds validation found with pattern " + group
    eval_ = {}
    for fold in folds:
        model = osp.join(fold, "best.model")
        name  = osp.basename(fold)
        ckpt  = torch.load(model)
        eval_[name] = [ckpt["eval"][k] for k in ckpt["eval"].keys()][:-1]
    out_df = pd.DataFrame.from_dict(eval_, orient= "index", columns = [ "acc", "tpr", "fpr"])
    out_df = out_df.append(out_df.mean(axis=0).rename("mean"))

    print("Mean Acc.:", out_df["acc"]["mean"])
    print("TPR      :", out_df["tpr"]["mean"])
    print("FPR      :", out_df["fpr"]["mean"])
    os.makedirs(args.out_dir, exist_ok = True)
    out_df.to_csv(osp.join(args.out_dir, "{}.csv".format(args.exp_name)))

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='nas-base-fold', help='name of the experiment')
    parser.add_argument('--log_dir', type=str, default='log', 
                        help='path to the log directory')
    parser.add_argument("--out_dir", type=str, default="eval_result", help='path for the output csv')
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    folds_evaluate(args)