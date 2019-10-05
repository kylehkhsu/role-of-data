#!/usr/bin/env bash
#SBATCH -o ./.slurm/slurm-%j.out
#SBATCH -p p100
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --array=0-63%64
#SBATCH -c 8
./scripts/wandb_sweep.sh