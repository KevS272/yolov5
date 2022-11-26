DEVICE=""
if [ -z "$1" ]
then
      DEVICE=cuda
else
      DEVICE=$1
fi

python3 modelswitcher.py \
    --data data/data_bothdn.yaml \
    --weightsmn ../w128.pth \
    --interpol 128 \
    --batch-size 1 \
    --weights best_d.pt \
    --weights2 best_n.pt \
    --task test \
    --device $DEVICE