#!/bin/bash

#SBATCH --job-name=01_sanity_check
#SBATCH --time=4:00:00
#SBATCH --partition=contrib-gpu
#SBATCH --output=experiments/09_predict_residuals/logs/01_sanity_check.out
#SBATCH --error=experiments/09_predict_residuals/logs/01_sanity_check.err
#SBATCH --exclude=gpu-g16,gpu-g28,gpu-g29,gpu-g38


echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"

source  ~/conda_init.sh
cd ~/projects/fourier_neural_operator/
conda activate fourier_neural_operator

python -m experiments.09_predict_residuals.train_models_sanity_check \
--data_fp /share/data/willett-group/meliao/data/2021-06-24_NLS_data_04_train.mat \
--test_data_fp /share/data/willett-group/meliao/data/2021-06-24_NLS_data_04_test.mat \
--emulator_fp experiments/08_FNO_pretraining/models/00_pretrain_ep_1000 \
--model_fp experiments/09_predict_residuals/models/01_sanity_check_ep_{} \
--train_df experiments/09_predict_residuals/results/01_sanity_check_train.txt \
--test_df experiments/09_predict_residuals/results/01_sanity_check_test.txt \
--freq_modes 8 \
--time_subsample 1 \
--epochs 1000
