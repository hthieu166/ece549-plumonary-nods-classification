#!/bin/bash
set -e


maxeps=9


for (( i=0; i<=$maxeps; i+=1)) 
do
    echo "process $i epoch"
    python main.py --batch_size 8 --fold $i --config_file configs/nas-model-2.yml
done 
