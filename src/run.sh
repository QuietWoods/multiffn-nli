#/bin/bash

python train.py --embeddings=/home/project/decomp-attn/glove/vectors.txt --train=/home/project/decomp-attn/data/atec_2.0_train.csv --validation=/home/project/decomp-attn/data/atec_2.0_dev.csv --save=saved-model --model='mlp' -e=400 --report=1000 
