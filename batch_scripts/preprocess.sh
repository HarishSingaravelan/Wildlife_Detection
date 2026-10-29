#!/bin/bash -l
# Configuration Options
#SBATCH --account=turbine
#SBATCH --partition=debug

#SBATCH --job-name=wildlife_adam
#SBATCH --output=batchlogs/preprocess_dataset_caribou_crt.out
#SBATCH --error=batchlogs/preprocess_dataset_caribou_crt.err
#SBATCH --mail-type=ALL
#SBATCH --time=0-12:00:00
#SBATCH --mem=256g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=34 #16

# Load Software
source /home/hs7569/dsci601/bin/activate

# Move to repo root
# cd /home/hs7569/github/DSCI-601-Wildlife || exit 1

export PYTHONPATH=$(pwd)


# Run training
# python data/debug_viz.py 

python data/preprocess_caribou.py

# python data/split_dataset.py \
#   --source_root "/shared/rc/turbine/Caribou/COCO_Dataset/images" \
#   --annotation_file "/shared/rc/turbine/Caribou/COCO_Dataset/caribou_coco_annotations.json" \
#   --output_root "/shared/rc/turbine/caribou_split_stratified" \
#   --copy_images
# python -m inference.manual_evaluate
