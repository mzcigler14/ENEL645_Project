#! /bin/bash
# ====================================
#SBATCH --job-name=DLProj-Run1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=log%j.txt
#SBATCH --error=error%j.txt
# ====================================

source ~/software/init-conda
conda activate pytorch

python Project1/main_project.py