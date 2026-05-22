#!/bin/bash
#SBATCH --job-name=abaqus_batch
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=10G
#SBATCH --array=1-1%1              
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
       mp_mode=threads \        # mpi and threads both ran within 3 mins of each other
       cpus=$SLURM_NTASKS \
       memory="9gb" \
       scratch=/mnt/scratch/$USER \
       ask_delete=OFF \
       interactive

#echo "Finished at: $(date)"

# print useful info to .out file - doesn't work, no output printed for memory size
echo ""
echo "----- Usage -----"
sleep 10 # wait 10s to allow cluster to update records
sacct -j "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
  --format=JobID,JobName%20,AllocCPUS,ReqMem,MaxRSS,MaxVMSize,AveRSS,Elapsed,State,ExitCode \
  --units=G


# ----------------- NOTES ----------------- #
# set N in --array=1-N%1 (line 7) to total number of .inp files listed in inpFiles.txt
# include this sbatch script in tar.gz
# - each .inp is treated as individual job (%1 means one at a time)
# - set per job (even if running more than one at once): --time, --nodes, --ntasks, --mem