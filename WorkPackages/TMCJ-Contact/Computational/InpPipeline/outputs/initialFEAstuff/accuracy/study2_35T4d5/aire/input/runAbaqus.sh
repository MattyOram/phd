# copy this into the directory containing the extracted tar.gz files
# set N in --array=1-N%1 to total number of .inp files listed in inpFiles.txt
# - each .inp is treated as individual job (%1 means one at a time)
# - set per job: --time, --nodes, --ntasks, --mem

#!/bin/bash
#SBATCH --job-name=abaqus_batch
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=8G
#SBATCH --array=1-5%1              
#SBATCH --output=abaqus_%A_%a.out
#SBATCH --error=abaqus_%A_%a.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=scmo@leeds.ac.uk

module load abaqus/2022

#export LM_LICENSE_FILE=port@host
export LM_LICENSE_FILE=27004@abaqus-research1.leeds.ac.uk #:$LM_LICENSE_FILE
export ABAQUSLM_LICENSE_FILE=$LM_LICENSE_FILE

cd $SLURM_SUBMIT_DIR

INPUT_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" inpFiles.txt) # get .inp file name from current line of inpFiles.txt
JOB_NAME="${INPUT_FILE%.inp}"                               # remove .inp from the file name


#echo "Abaqus job name: $JOB_NAME"
#echo "Task ID: $SLURM_ARRAY_TASK_ID"
#echo "Started at: $(date)"

abaqus job=$JOB_NAME \
       mp_mode=threads \
       cpus=$SLURM_NTASKS \
       scratch=/mnt/scratch/$USER \
       ask_delete=OFF \
       interactive

#echo "Finished at: $(date)"