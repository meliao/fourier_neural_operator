#!/bin/bash

#SBATCH --job-name=GRF1
#SBATCH --partition=contrib-gpu
#SBATCH --output=/home-nfs/meliao/projects/fourier_neural_operator/logs/GRF1_fourier_1d.out
#SBATCH --error=/home-nfs/meliao/projects/fourier_neural_operator/logs/GRF1_fourier_1d.err
echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"

source ~/conda_init.sh
conda activate basis_emulators

python fourier_1d.py \
--data_fp data/2021-03-17_training_Burgers_data_GRF1.mat \
--model_fp models/GRF1_fourier_1d_model \
--preds_fp preds/GRF1_fourier_1d_preds.mat
