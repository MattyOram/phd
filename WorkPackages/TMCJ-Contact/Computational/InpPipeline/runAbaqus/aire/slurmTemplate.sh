#!/bin/bash
#SBATCH --job-name=35T-neutral-04
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=8G
#SBATCH --output=abaqus_%j.out
#SBATCH --error=abaqus_%j.err

module load abaqus/2022

export LM_LICENSE_FILE=port@host
export ABAQUSLM_LICENSE_FILE=$LM_LICENSE_FILE

cd $SLURM_SUBMIT_DIR

abaqus job=model \
       mp_mode=mpi \
       cpus=$SLURM_NTASKS \
       scratch=/mnt/scratch/$USER \
       ask_delete=OFF \
       interactive 

