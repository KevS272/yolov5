#!/bin/sh 
DEVICE=""
if [ "$USER" = "sabolc" ]
then
      DEVICE=cpu
else
      DEVICE=cuda:0
fi

echo using $DEVICE

python3 val.py \
    --data data/data_viz.yaml \
    --batch-size 1 \
    --weights best_d.pt best_n.pt \
    --task test \
    --name "ensemble" \
    --device $DEVICE