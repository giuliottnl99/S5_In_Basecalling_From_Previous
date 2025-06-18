#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --ntasks-per-node=32
#SBATCH --partition=cuda
#SBATCH --gres=gpu:4
#SBATCH --job-name=prova_gu_resume
#SBATCH --mem=120GB
#SBATCH --mail-type=ALL
#SBATCH --nodes=1

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



python /home/mla_group_17/group_11/mla-prj-24-mla24-prj17-gu3/sbonito/scripts/train_original.py \
--data-dir /home/mla_group_17/group_11/mla-prj-24-mla24-prj17-gu3/preprocessed_data \
--output-dir /home/mla_group_17/group_11/mla-prj-24-mla24-prj17-gu3/deep_fast_prova \
--model bonitosnn \
--window-size 2000 \
--batch-size 64 \
--starting-lr 0.01 \
--nlstm 0 \
--num-epochs 5 \
--overwrite \
--nl2mu 5

