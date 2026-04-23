#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=320G
#SBATCH --output=slurm_logs/output-%j.out
#SBATCH --error=slurm_logs/error-%j.err
#SBATCH --mail-type ALL          # Events: BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user yaoyin@uw.edu   # Optional — defaults to your cluster account email

source ~/.bash_profile
module load miniconda
conda activate /nfs/roberts/scratch/pi_kmh66/yy777/ycrc_conda/envs/ssm
cd /home/yy777/project8/ssm/neutrino_project
python cli.py fit --config $1 ${2:+--ckpt_path $2}
