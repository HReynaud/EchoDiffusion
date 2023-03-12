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

export model=/PATH/TO/TRAINED/MODEL
export chunks=8

for gpu in $(seq 0 $((chunks-1)))
do
    echo "Starting on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python diffusion/generate_samples/generate_dataset.py --model $model --bs 4 --chunks $chunks --chunk $gpu --ef_list diffusion/generate_samples/balanced_ef_list.csv --cond_scale 1 &
done

python diffusion/generate_samples/merge_dataset_reports.py --model $model --chunks $chunks --del_prev