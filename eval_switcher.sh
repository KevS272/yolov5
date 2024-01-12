#!/bin/sh 
DEVICE=""
if [ "$USER" = "sabolc" ]
then
      DEVICE=cpu
else
      DEVICE=cuda:0
fi

echo using $DEVICE

python3 modelswitcher.py \
    --data data/data_viz.yaml \
    --weightsmn ../w128.pth \
    --interpol 128 \
    --batch-size 1 \
    --weights best_d.pt \
    --weights2 best_n.pt \
    --task test \
    --name "switcher" \
    --device $DEVICE