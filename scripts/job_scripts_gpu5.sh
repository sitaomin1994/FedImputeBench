#!/bin/bash
#SBATCH --job-name=eval_hhip
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sm2370@rutgers.edu
#SBATCH --partition=p_jsvaidya_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-cpu=7G
#SBATCH --time=3-00:00:00
#SBATCH --nodelist=gpuk001,gpuk002

#SBATCH --export=ALL

# Output and Error File Names
#SBATCH --output=./slurm/%N.%j.out  # STDOUT output file
#SBATCH --error=./slurm/%N.%j.err   # STDERR output file
source /projects/community/miniconda3/bin/activate impute4fair
cd /scratch/sm2370/fedmiss_bench/

# Your command here
srun ./scripts/gpu5.sh