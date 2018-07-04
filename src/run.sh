#/bin/bash

python predict.py saved_model glove/vectors.txt -i -inputfile $1 -outputfile $2

