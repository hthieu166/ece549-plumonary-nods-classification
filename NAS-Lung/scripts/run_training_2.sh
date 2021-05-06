#!/bin/bash
set -e
CONFIG=dpn3d
CUDA_VISIBLE_DEVICES=0
maxeps=5

for (( i=5; i<=$maxeps; i+=1)) 
do
    echo "process $i epoch"
    python main.py --fold $i --config_file configs/$CONFIG.yml
done 

python code/folds_evaluate.py --exp_name $CONFIG-fold
