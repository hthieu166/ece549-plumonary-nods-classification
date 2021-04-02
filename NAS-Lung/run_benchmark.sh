#!/bin/bash
set -e
CONFIG=nas-model-1-angle-expr-1
CUDA_VISIBLE_DEVICES=1
echo "Running 5-5 folds strategy"
python main.py \
    --eval_mode 5fold --config_file configs/$CONFIG.yml
