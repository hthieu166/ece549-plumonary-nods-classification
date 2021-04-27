#!/bin/bash
set -e
CONFIG=multi-views-expr-4
maxeps=9

for (( i=0; i<=$maxeps; i+=1)) 
do
    echo "process $i epoch"
    CUDA_VISIBLE_DEVICES=1 python main.py --fold $i --config_file configs/$CONFIG.yml \
    --savemodel log/-fold-5 --mode test
done 

python code/folds_evaluate.py --exp_name $CONFIG-fold
