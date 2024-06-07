#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=24:00:00

python train.py --gin_path "configs/vcc2018.gin" --save_path "runs/test_vcc2018"