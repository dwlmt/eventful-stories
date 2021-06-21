#!/usr/bin/env bash
#SBATCH -o /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -e /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=16g  # Memory
#SBATCH --cpus-per-task=8  # number of cpus to use - there are 32 on each node.

echo "============"
echo "Initialize Env ========"

set -e # fail fast

export CURRENT_TIME=$(date "+%Y_%m_%d_%H%M%S")

# Activate Conda
source /home/${USER}/miniconda3/bin/activate allennlp2

echo "I'm running on ${SLURM_JOB_NODELIST}"
dt=$(date '+%d_%m_%y__%H_%M')
echo ${dt}

# Env variables
export STUDENT_ID=${USER}

# General training parameters
export CLUSTER_HOME="/home/${STUDENT_ID}"

declare -a ScratchPathArray=(/disk/scratch_big/ /disk/scratch1/ /disk/scratch2/ /disk/scratch/ /disk/scratch_fast/)

# Iterate the string array using for loop
for i in "${ScratchPathArray[@]}"; do
  echo ${i}
  if [ -d ${i} ]; then
    export SCRATCH_HOME="${i}/${STUDENT_ID}"
    mkdir -p ${SCRATCH_HOME}
    if [ -w ${SCRATCH_HOME} ]; then
      break
    fi
  fi
done

echo ${SCRATCH_HOME}

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
export LINE=$(sed "${SLURM_ARRAY_TASK_ID}q;d" ${CLUSTER_HOME}/${BATCH_FILE_PATH}/${BATCH_FILE_NAME})
export PREDICTION_STORY_FILE="${CLUSTER_HOME}/${BATCH_FILE_PATH}/${LINE}"

export EXP_ROOT="${CLUSTER_HOME}/git/story-fragments"
export EXP_ID="${EXP_NAME}_${SLURM_JOB_ID}_${CURRENT_TIME}"
export SERIAL_DIR="${SCRATCH_HOME}/${EXP_ID}/"

# Ensure the scratch home exists and CD to the experiment root level.
cd "${EXP_ROOT}" # helps AllenNLP behave

cd /home/s1569885/git/eventful-stories/eventful_stories/data_processing

python coref_srl_dataset_processing.py process \
--data-file ${PREDICTION_STORY_FILE} \
--output-file ${SERIAL_DIR} \
--script-path ${SCRIPT_PATH} \
--dataset-name ${DATASET_NAME}

echo "============"
echo "Dataset processing finished"

export HEAD_EXP_DIR="${CLUSTER_HOME}/runs/${EXP_ID}"
mkdir -p "${HEAD_EXP_DIR}"
rsync -avuzhP "${SERIAL_DIR}/" "${HEAD_EXP_DIR}/" # Copy output onto headnode

rm -rf "${SERIAL_DIR}"

echo "============"
echo "results synced"