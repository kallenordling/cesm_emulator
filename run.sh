#!/bin/bash
#SBATCH --account=project_2014946
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:4
#SBATCH --time=70:00:00
#SBATCH --mem=64G
#SBATCH --job-name=diffusion-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Load required modules
module purge
module load tykky
module load pytorch

# Optional: activate virtual environment if you have one
# source /projappl/project_2014946/venvs/myenv/bin/activate

# Run your training script
bash train.sh
