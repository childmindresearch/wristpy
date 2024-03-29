#!/bin/bash

#SBATCH --job-name=wristpy_processing
#SBATCH --array=1
#SBATCH --output=/ocean/projects/med220004p/shared/data_raw/backup_onprem/adam/external_ID/logs/job_%A_%a.out
#SBATCH --error=/ocean/projects/med220004p/shared/data_raw/backup_onprem/adam/external_ID/logs/job_%A_%a.err
#SBATCH --nodes 1
#SBATCH --partition RM-shared
#SBATCH --time 24:00:00
#SBATCH --ntasks-per-node 8
set -x
# Your singularity container path
SINGULARITY_CONTAINER=/ocean/projects/med220004p/shared/data_raw/backup_onprem/adam/wristpy_process.sif 

# Base directory and output dir for data
BASE_DIR=/ocean/projects/med220004p/shared/data_raw/backup_onprem/adam/external_ID/
OUTPUT_DIR=/ocean/projects/med220004p/shared/data_raw/backup_onprem/adam/wristpy_out/


# Run the singularity container on the sub-directory
# Bind the BASE_DIR to both /data and output dir /output in the container

singularity run --bind ${BASE_DIR}:/app/data --bind ${OUTPUT_DIR}:/app/output ${SINGULARITY_CONTAINER} 