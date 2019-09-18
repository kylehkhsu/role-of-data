#!/usr/bin/env bash
#srun --gres=gpu:1 -c 8 --mem=12G -p p100
wandb login d34e5033d016b5e8f34841d773d02b48cbd74cc7
wandb on
wandb agent 24eutu6n