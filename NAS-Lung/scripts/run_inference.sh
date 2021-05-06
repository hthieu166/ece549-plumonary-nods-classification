#!/bin/bash
set -e
# CONFIG=nas-model-1-angle-expr-4
CONFIG=multi-views-expr-4
maxeps=5

for (( i=1; i<=$maxeps; i+=1)) 
do
    echo "process $i epoch"
    CUDA_VISIBLE_DEVICES=0 python main.py --fold $i --config_file configs/$CONFIG.yml \
    --savemodel log/$CONFIG-fold-$i/best.model --resume --mode infer
done
