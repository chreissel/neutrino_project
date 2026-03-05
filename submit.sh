#!/bin/bash
#SBATCH --partition=iaifi_gpu
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=320G
#SBATCH --output=slurm_logs/output-%j.out
#SBATCH --error=slurm_logs/error-%j.err

source ~/.bash_profile
conda activate ssm
cd /n/home03/creissel/neutrino_project/
if [ -n "$2" ]; then
    python cli.py fit --config $1 --ckpt_path $2
else
    python cli.py fit --config $1
fi
