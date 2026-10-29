#!/bin/bash -l
# Configuration Options
#SBATCH --account=turbine
#SBATCH --partition=debug



#SBATCH --job-name=wildlife_adam
#SBATCH --output=batchlogs/detr_caribou/detr_adaptive.out
#SBATCH --error=batchlogs/detr_caribou/detr_adaptive.err
#SBATCH --mail-type=ALL
#SBATCH --time=1-00:00:00 
#SBATCH --mem=256g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=34 #34
#SBATCH --gres=gpu:a100:1

# Load Software
source /home/hs7569/dsci601/bin/activate

# Move to repo root
# cd /home/hs7569/github/DSCI-601-Wildlife || exit 1

export PYTHONPATH=$(pwd)
export PYTHONDONTWRITEBYTECODE=1

# Run training
python3 modeling/detr/train_detr.py
# python3 inference/manual_evaluate.py
