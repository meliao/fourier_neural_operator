#!/bin/bash

#SBATCH --job-name=00_no_backprop
#SBATCH --time=4:00:00
#SBATCH --partition=contrib-gpu-long
#SBATCH --output=experiments/11_PDE_Loss/logs/00_no_backprop.out
#SBATCH --error=experiments/11_PDE_Loss/logs/00_no_backprop.err
#SBATCH --exclude=gpu-g16,gpu-g28,gpu-g29,gpu-g38


echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"

source  ~/conda_init.sh
cd ~/projects/fourier_neural_operator/
conda activate fourier_neural_operator

python -m experiments.11_PDE_Loss.train_models_discrete_PDE_loss \
--data_fp /share/data/willett-group/meliao/data/2021-06-24_NLS_data_04_train.mat \
--test_data_fp /share/data/willett-group/meliao/data/2021-06-24_NLS_data_04_test.mat \
--model_fp experiments/11_PDE_Loss/models/00_no_backprop_ep_{} \
--train_df experiments/11_PDE_Loss/results/00_no_backprop_train.txt \
--test_df experiments/11_PDE_Loss/results/00_no_backprop_test.txt \
--lr_exp "-3" \
--freq_modes 8 \
--time_subsample 1 \
--epochs 1000 \
