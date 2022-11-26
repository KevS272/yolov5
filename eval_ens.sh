#!/bin/sh 
DEVICE=""
if [ "$USER" = "sabolc" ]
then
      DEVICE=cpu
else
      DEVICE=cuda
fi

echo using $DEVICE

python3 val.py \
    --data data/data_bothdn.yaml \
    --batch-size 1 \
    --weights best_d.pt \
    --weights2 best_n.pt \
    --task test \
    --name "ensemble" \
    --device $DEVICE