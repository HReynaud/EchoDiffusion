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
    CUDA_VISIBLE_DEVICES=$gpu python diffusion/evaluate/generate_score_file_chunk.py --model $model --bs 4 --num_noise 3 --chunks $chunks --chunk $gpu --save_videos --rand_ef &
done

python diffusion/evaluate/merge_score_files.py --model $model --chunks $chunks --del_prev --rand_ef