#!/bin/bash
nohup python3 -u train.py --id 1 --seed 0 --prune_k -1 --lr 0.5 --rnn_hidden 200 --num_epoch 150 --pooling max --mlp_layers 2 --pooling_l2 0.003 > out.log 2>&1 &
