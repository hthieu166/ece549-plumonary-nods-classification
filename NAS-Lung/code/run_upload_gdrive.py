import os
import glob
import os.path as osp
import argparse
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='nas-base-fold', help='name of the experiment')
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    folds = sorted(glob.glob(osp.join("log", args.exp_name + "*")))
    for fold in folds:
        os.system("scripts/upload_gdrive.sh " + fold)