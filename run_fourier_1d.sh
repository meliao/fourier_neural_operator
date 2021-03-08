#!/bin/bash

#SBATCH --job-name=fourier_1d_small_data
#SBATCH --partition=contrib-gpu
#SBATCH --output=/home-nfs/meliao/projects/fourier_neural_operator/fourier_1d_small_data.out
#SBATCH --error=/home-nfs/meliao/projects/fourier_neural_operator/fourier_1d_small_data.err
echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"

source ~/conda_init.sh
conda activate basis_emulators

python fourier_1d.py \
--data_fp data/burgers_data_R10.mat \
--model_fp models/fourier_1d_model \
--preds_fp preds/fourier_1d_preds.mat
