#!/bin/bash
set -e
CONFIG=multi-views-expr-4-nlst
maxeps=5

for (( i=1; i<=$maxeps; i+=1)) 
do
    echo "process $i epoch"
    CUDA_VISIBLE_DEVICES=0 python main.py --fold $i --config_file configs/$CONFIG.yml \
    --savemodel /mnt/data0-nfs/anhleu2/log/$CONFIG-fold-$i/best.model --resume --mode infer --nlst True
done
