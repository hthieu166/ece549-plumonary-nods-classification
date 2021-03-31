#!/bin/bash
set -e
CONFIG=se-res-model-1
maxeps=9

for (( i=0; i<=$maxeps; i+=1)) 
do
    echo "process $i epoch"
    CUDA_VISIBLE_DEVICES=0 python main.py --fold $i --config_file configs/$CONFIG.yml
done 

python code/folds_evaluate.py --exp_name $CONFIG-fold
