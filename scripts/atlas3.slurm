#!/bin/bash
#SBATCH --job-name=atlas3
#SBATCH --mem=48G
#SBATCH -t 2-00:00:00
#SBATCH -c 8
#SBATCH -N 1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH --gres=gpu:1
#SBATCH -D /cluster/projects/radiomics/Temp/sejin/AtlasNet
#SBATCH --mail-user=sejin.kim@uhnresearch.ca
#SBATCH --mail-type=END
#SBATCH --output="/cluster/projects/radiomics/Temp/sejin/slurm_logs/%A-%x.out"

USERNAME="sejinkim"
source /cluster/home/$USERNAME/.bashrc
conda activate atlas3

python train.py --shapenet13 --nb_primitives 9 --template_type "SQUARE" 
