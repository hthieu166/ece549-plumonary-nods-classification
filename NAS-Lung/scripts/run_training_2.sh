#!/bin/bash
set -e
CONFIG=multi-views-expr-5
CUDA_VISIBLE_DEVICES=0
maxeps=5

for (( i=1; i<=$maxeps; i+=1)) 
do
    echo "process $i epoch"
    python main.py --fold $i --config_file configs/$CONFIG.yml
done 

python code/folds_evaluate.py --exp_name $CONFIG-fold
