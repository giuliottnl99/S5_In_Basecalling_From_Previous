#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --job-name=prova_gu_demo_data
#SBATCH --mem=10GB
#SBATCH --mail-type=ALL

# ---------------------------------------------------------------------------- #
#                            SLURM Batch Script                                #
# ---------------------------------------------------------------------------- #

# Load necessary modules for CUDA and Python
echo "Loading CUDA and Python modules..."
module load nvidia/cudasdk/10.1
module load intel/python/3

# ---------------------------------------------------------------------------- #


cd /home/mla_group_17/group_11
pwd 
source .bashrc

echo "Activating the Conda environment..."
conda activate 

# Run the training script
python /home/mla_group_17/group_11/mla-prj-24-mla24-prj17-gu3/sbonito/scripts/train_original.py \
--data-dir /home/mla_group_17/group_11/mla-prj-24-mla24-prj17-gu3/preprocessed_data \
--output-dir /home/mla_group_17/group_11/mla-prj-24-mla24-prj17-gu3/one_one \
--model bonitosnn \
--window-size 2000 \
--batch-size 64 \
--starting-lr 0.1 \
--nlstm 0 \
--checkpoint /home/mla_group_17/group_11/mla-prj-24-mla24-prj17-gu3/one_one/checkpoints/checkpoint_60000.pt \
--num-epochs 10 \
--overwrite \
--nl2mu 1 \
--one-conv True