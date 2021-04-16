#!/bin/bash
#SBATCH --job-name=atlasnet
#SBATCH --mem=48G
#SBATCH -t 2-00:00:00
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -p himem
#SBATCH -D /cluster/projects/radiomics/Temp/sejin/AtlasNet
#SBATCH --mail-user=sejin.kim@uhnresearch.ca
#SBATCH --mail-type=END
#SBATCH --output="/cluster/projects/radiomics/Temp/sejin/slurm_logs/%A-%x.out"

USERNAME="sejinkim"
source /cluster/home/$USERNAME/.bashrc
rm /cluster/projects/radiomics/Temp/sejin/AtlasNet/dataset/data/ShapeNetV1Renderings -r
