#!/bin/bash
set -e
CONFIG=multi-views-fusion-v1
maxeps=5

for (( i=1; i<=$maxeps; i+=1)) 
do
    echo "process $i epoch"
    CUDA_VISIBLE_DEVICES=0 python main.py --fold $i --config_file configs/$CONFIG.yml
done 

python code/folds_evaluate.py --exp_name $CONFIG-fold
