#!/bin/bash

#SBATCH --job-name=01_test_job
#SBATCH --time=4:00:00
#SBATCH --partition=contrib-gpu
#SBATCH --output=experiments/06_time_dependent_models/logs/01_test_job.out
#SBATCH --error=experiments/06_time_dependent_models/logs/01_test_job.err

echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"

source  ~/conda_init.sh
cd ~/projects/fourier_neural_operator/
conda activate fourier_neural_operator

python -m experiments.06_time_dependent_models.train_models \
--data_fp data/2021-06-10_NLS_data/NLS_data_seed_0.mat \
--test_data_fp data/2021-06-10_NLS_data/NLS_data_seed_1.mat \
--model_fp experiments/06_time_dependent_models/models/test_model_01 \
--results_fp experiments/06_time_dependent_models/results/test_results.txt \
--train_df experiments/06_time_dependent_models/results/test_train_01.txt \
--freq_modes 8 \
--time_subsample 10
