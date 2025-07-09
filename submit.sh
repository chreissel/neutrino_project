#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/output-%j.out
#SBATCH --error=slurm_logs/error-%j.err

source ~/.bash_profile
mamba activate ssm
cd /n/home03/creissel/neutrino_project/
python cli.py fit --config $1
