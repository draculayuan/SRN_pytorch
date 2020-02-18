#!/bin/bash

ROOT=..
export PYTHONPATH=$ROOT:$PYTHONPATH

python -u train.py \
  --config=$ROOT/config/config.json \
  --img_list=$ROOT/image_list.txt \
  --resume=$ROOT/model/SRN.pth \
  --max_size=2100 \
  2>&1 | tee test.log

