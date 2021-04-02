#!/bin/bash
set -e
CONFIG=nas-model-1-author
maxeps=9

for (( i=0; i<=$maxeps; i+=1)) 
do
    echo "process $i epoch"
    CUDA_VISIBLE_DEVICES=1 python main.py --fold $i --config_file configs/$CONFIG.yml \
    --savemodel log/ckpt/Model-1/checkpoint-5/ckpt.t7 --resume --mode test
done 

python code/folds_evaluate.py --exp_name $CONFIG-fold
