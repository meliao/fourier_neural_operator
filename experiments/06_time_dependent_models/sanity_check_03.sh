#!/bin/bash

#SBATCH --job-name=03_sanity_check
#SBATCH --time=4:00:00
#SBATCH --partition=contrib-gpu
#SBATCH --output=experiments/06_time_dependent_models/logs/03_sanity_check.out
#SBATCH --error=experiments/06_time_dependent_models/logs/03_sanity_check.err

echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"

source  ~/conda_init.sh
cd ~/projects/fourier_neural_operator/
conda activate fourier_neural_operator


python -m experiments.06_time_dependent_models.train_models_sanity_check_1 \
--data_fp ~/projects/fourier_neural_operator/data/2021-06-21_NLS_one_step_dataset_train.mat \
--test_data_fp ~/projects/fourier_neural_operator/data/2021-06-21_NLS_one_step_dataset_test.mat \
--train_df ~/projects/fourier_neural_operator/experiments/06_time_dependent_models/results/03_sanity_check.txt \
--model_fp ~/projects/fourier_neural_operator/experiments/06_time_dependent_models/models/03_sanity_check \
--results_fp ~/projects/fourier_neural_operator/experiments/06_time_dependent_models/second_sanity_check.txt \
--freq_modes 8 \
--time_subsample 1 \
--epochs 500
