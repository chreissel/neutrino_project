#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=slurm_logs/output-%j.out
#SBATCH --error=slurm_logs/error-%j.err

source ~/.bash_profile
module load miniconda
conda activate ssm
cd /home/lfd34/project/project8/ssm/neutrino_project
python cli.py fit --config $1
