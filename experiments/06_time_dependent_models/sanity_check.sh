#!/bin/bash

#SBATCH --job-name=00_sanity_check
#SBATCH --time=4:00:00
#SBATCH --partition=contrib-gpu
#SBATCH --output=experiments/06_time_dependent_models/logs/00_sanity_check.out
#SBATCH --error=experiments/06_time_dependent_models/logs/00_sanity_check.err

echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"

source  ~/conda_init.sh
cd ~/projects/fourier_neural_operator/
conda activate fourier_neural_operator


python -m experiments.06_time_dependent_models.train_models_sanity_check \
--data_fp ~/projects/fourier_neural_operator/data/2021-06-18_NLS_one_step_train.mat \
--test_data_fp ~/projects/fourier_neural_operator/data/2021-06-18_NLS_one_step_test.mat \
--train_df ~/projects/fourier_neural_operator/experiments/06_time_dependent_models/results/01_sanity_check.txt \
--model_fp ~/projects/fourier_neural_operator/experiments/06_time_dependent_models/models/01_sanity_check \
--results_fp ~/projects/fourier_neural_operator/experiments/06_time_dependent_models/01_sanity_check.txt \
--freq_modes 8 \
--time_subsample 1 \
--epochs 2
