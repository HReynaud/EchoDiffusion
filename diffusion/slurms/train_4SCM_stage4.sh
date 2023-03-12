#!/bin/bash -l
#
#SBATCH --time=24:00:00
#SBATCH --job-name=diffusion_echo
#SBATCH --mail-type=ALL
#SBATCH --export=NONE
#SBATCH --mail-user=###
#SBATCH --output=slurm_%j.log
#SBATCH --gres=gpu:a100:8
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH -C a100_80


# ADD YOUR SLURM ENV SETTINGS HERE #

wandb online

timeout 23h accelerate launch --multi_gpu --num_processes=8 diffusion/train.py --config diffusion/configs/4SCM.yaml --stage 4 --bs 5 --ignore_time 0.25

if [[ $? -eq 124 ]]; then
    sbatch diffusion/slurms/train_4SCM_stage4.sh
