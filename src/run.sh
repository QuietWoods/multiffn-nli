#/bin/bash

python train.py --embeddings=glove/vectors.txt --train=data/train_90000.csv --validation=data/dev_12477.csv --save=saved-model --model='lstm' -e=400 --report=1000
