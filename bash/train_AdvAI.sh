#!/usr/bin/env bash
export CUDA_NUMBER=0
CUDA_VISIBLE_DEVICES=$CUDA_NUMBER python train_AdvAI.py --prior=uniform --beta=0.95 \
--int_hidden_units=100 --int_hidden_layers=3 --disc_hidden_layers=3 \
--n_scale=10.0 --use_ema=1 --dims=3



