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
    --weights best_dn.pt \
    --task test \
    --name "mono" \
    --device $DEVICE