#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate echosyn

echo "Using conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
# Script to run all metrics calculations sequentially
# Continues to next task even if one fails
echo "Starting metrics calculation for all datasets..."

# ========================================
# PSAX Triplets (GPU 4)
# ========================================
echo "Starting PSAX triplets calculations on GPU 4..."

echo "Task 1: PSAX Full dataset metrics..."
CUDA_VISIBLE_DEVICES='4' python src/scripts/calc_metrics_for_dataset.py \
    --real_data_path /nfs/usrhome/khmuhammad/EchoPath/data/reference/ped_psax \
    --fake_data_path /nfs/usrhome/khmuhammad/EchoPath/samples/lvdm_psax_triplets/jpg \
    --mirror 0 --gpus 1 --resolution 112 \
    --metrics fvd2048_16f,fid50k_full,is50k \
    >> "/nfs/usrhome/khmuhammad/EchoPath/samples/lvdm_psax_triplets/full_metrics.txt"

echo "Task 2: PSAX Good dataset metrics..."
CUDA_VISIBLE_DEVICES='4' python src/scripts/calc_metrics_for_dataset.py \
    --real_data_path /nfs/usrhome/khmuhammad/EchoPath/data/reference/ped_psax \
    --fake_data_path /nfs/scratch/EchoPath/samples/lvdm_psax_triplets/good_jpg \
    --mirror 0 --gpus 1 --resolution 112 \
    --metrics fvd2048_16f,fid50k_full,is50k \
    >> "/nfs/scratch/EchoPath/samples/lvdm_psax_triplets/good_metrics.txt"

echo "Task 3: PSAX Bad dataset metrics..."
CUDA_VISIBLE_DEVICES='4' python src/scripts/calc_metrics_for_dataset.py \
    --real_data_path /nfs/usrhome/khmuhammad/EchoPath/data/reference/ped_psax \
    --fake_data_path /nfs/scratch/EchoPath/samples/lvdm_psax_triplets/bad_jpg \
    --mirror 0 --gpus 1 --resolution 112 \
    --metrics fvd2048_16f,fid50k_full,is50k \
    >> "/nfs/scratch/EchoPath/samples/lvdm_psax_triplets/bad_metrics.txt"

# ========================================
# A4C Triplets (GPU 5)
# ========================================
echo "Starting A4C triplets calculations on GPU 5..."

echo "Task 4: A4C Full dataset metrics..."
CUDA_VISIBLE_DEVICES='5' python src/scripts/calc_metrics_for_dataset.py \
    --real_data_path /nfs/usrhome/khmuhammad/EchoPath/data/reference/ped_a4c \
    --fake_data_path /nfs/usrhome/khmuhammad/EchoPath/samples/lvdm_a4c_triplets/jpg \
    --mirror 0 --gpus 1 --resolution 112 \
    --metrics fvd2048_16f,fid50k_full,is50k \
    >> "/nfs/usrhome/khmuhammad/EchoPath/samples/lvdm_a4c_triplets/full_metrics.txt"

echo "Task 5: A4C Good dataset metrics..."
CUDA_VISIBLE_DEVICES='5' python src/scripts/calc_metrics_for_dataset.py \
    --real_data_path /nfs/usrhome/khmuhammad/EchoPath/data/reference/ped_a4c \
    --fake_data_path /nfs/scratch/EchoPath/samples/lvdm_a4c_triplets/good_jpg \
    --mirror 0 --gpus 1 --resolution 112 \
    --metrics fvd2048_16f,fid50k_full,is50k \
    >> "/nfs/scratch/EchoPath/samples/lvdm_a4c_triplets/good_metrics.txt"

echo "Task 6: A4C Bad dataset metrics..."
CUDA_VISIBLE_DEVICES='5' python src/scripts/calc_metrics_for_dataset.py \
    --real_data_path /nfs/usrhome/khmuhammad/EchoPath/data/reference/ped_a4c \
    --fake_data_path /nfs/scratch/EchoPath/samples/lvdm_a4c_triplets/bad_jpg \
    --mirror 0 --gpus 1 --resolution 112 \
    --metrics fvd2048_16f,fid50k_full,is50k \
    >> "/nfs/scratch/EchoPath/samples/lvdm_a4c_triplets/bad_metrics.txt"

# ========================================
# ASD Triplets (GPU 1)
# ========================================
echo "Starting ASD triplets calculations on GPU 1..."

echo "Task 7: ASD Full dataset metrics..."
CUDA_VISIBLE_DEVICES='4' python src/scripts/calc_metrics_for_dataset.py \
    --real_data_path /nfs/usrhome/khmuhammad/EchoPath/data/reference/cardiac_asd \
    --fake_data_path /nfs/usrhome/khmuhammad/EchoPath/samples/lvdm_asd_triplets_2000/jpg \
    --mirror 0 --gpus 1 --resolution 112 \
    --metrics fvd2048_16f,fid50k_full,is50k \
    >> "/nfs/usrhome/khmuhammad/EchoPath/samples/lvdm_asd_triplets_2000/full_metrics.txt"

echo "Task 8: ASD Good dataset metrics..."
CUDA_VISIBLE_DEVICES='1' python src/scripts/calc_metrics_for_dataset.py \
    --real_data_path /nfs/usrhome/khmuhammad/EchoPath/data/reference/cardiac_asd \
    --fake_data_path /nfs/scratch/EchoPath/samples/lvdm_asd_triplets_2000/good_jpg \
    --mirror 0 --gpus 1 --resolution 112 \
    --metrics fvd2048_16f,fid50k_full,is50k \
    >> "/nfs/scratch/EchoPath/samples/lvdm_asd_triplets_2000/good_metrics.txt"

echo "Task 9: ASD Bad dataset metrics..."
CUDA_VISIBLE_DEVICES='1' python src/scripts/calc_metrics_for_dataset.py \
    --real_data_path /nfs/usrhome/khmuhammad/EchoPath/data/reference/cardiac_asd \
    --fake_data_path /nfs/scratch/EchoPath/samples/lvdm_asd_triplets_2000/bad_jpg \
    --mirror 0 --gpus 1 --resolution 112 \
    --metrics fvd2048_16f,fid50k_full,is50k \
    >> "/nfs/scratch/EchoPath/samples/lvdm_asd_triplets_2000/bad_metrics.txt"

echo "All metrics calculations completed!"
echo "Check the respective output files for results."