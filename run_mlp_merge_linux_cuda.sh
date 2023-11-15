#!/bin/bash

DEVICE="cuda"
D0="FashionMNIST"
D1="MNIST"

python mlp_merge.py --device $DEVICE --dataset0 $D0 --dataset1 $D1