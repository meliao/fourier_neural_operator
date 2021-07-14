#!/bin/bash

#SBATCH --job-name=00_one_step
#SBATCH --time=4:00:00
#SBATCH --partition=contrib-gpu
#SBATCH --output=experiments/10_linear_approx/logs/00_one_step.out
#SBATCH --error=experiments/10_linear_approx/logs/00_one_step.err

echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"

source  ~/conda_init.sh
cd ~/projects/fourier_neural_operator/
conda activate fourier_neural_operator


python -m experiments.10_linear_approx.train_models \
--data_fp /share/data/willett-group/meliao/data/2021-07-144_NLS_data_05_train.mat \
--test_data_fp /share/data/willett-group/meliao/data/2021-07-14_NLS_data_05_test.mat \
--epochs 1000 \
--time_subsample 1 \
--train_df ~/projects/fourier_neural_operator/experiments/10_linear_approx/results/00_one_step_train.txt \
--test_df ~/projects/fourier_neural_operator/experiments/10_linear_approx/results/00_one_step_test.txt \
--model_fp ~/projects/fourier_neural_operator/experiments/10_linear_approx/models/00_one_step_ep_{}