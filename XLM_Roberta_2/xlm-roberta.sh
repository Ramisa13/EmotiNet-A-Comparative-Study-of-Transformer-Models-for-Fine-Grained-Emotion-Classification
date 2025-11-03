#!/bin/bash
#SBATCH --job-name=robert-emotion
#SBATCH --output=robert_output_%j.txt
#SBATCH --error=robert_error_%j.txt
#SBATCH --partition=gpusmall
#SBATCH --account=project_2011211
#SBATCH --gres=gpu:a100:1,nvme:200
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=12:00:00

module load pytorch/2.7

export LOCAL_SCRATCH=${LOCAL_SCRATCH:-/tmp}
export HF_HOME=$LOCAL_SCRATCH/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export HF_METRICS_CACHE=$HF_HOME

echo "Job started on $(hostname) at $(date)"
echo "Using LOCAL_SCRATCH: $LOCAL_SCRATCH"
echo "HF cache dirs: $HF_HOME"

yes | python3 /scratch/project_2011211/Fahim/affective_computing/xlm-roberta.py

echo "Job ended at $(date)"
