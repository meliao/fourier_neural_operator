#!/bin/bash

#SBATCH --job-name=00_test_job
#SBATCH --time=4:00:00
#SBATCH --partition=contrib-gpu
#SBATCH --output=experiments/07_long_time_dependent_runs/logs/01_baseline.out
#SBATCH --error=experiments/07_long_time_dependent_runs/logs/01_baseline.err
#SBATCH --exclude=gpu-g16,gpu-g28,gpu-g29,gpu-g38

echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"

source  ~/conda_init.sh
cd ~/projects/fourier_neural_operator/
conda activate fourier_neural_operator

python -m experiments.07_long_time_dependent_runs.train_models_baseline \
--data_fp /share/data/willett-group/meliao/data/2021-06-24_NLS_data_04_train.mat \
--test_data_fp /share/data/willett-group/meliao/data/2021-06-24_NLS_data_04_test.mat \
--model_fp experiments/07_long_time_dependent_runs/models/01_baseline_ep_{} \
--train_df experiments/07_long_time_dependent_runs/results/01_baseline_train.txt \
--test_df experiments/07_long_time_dependent_runs/results/01_baseline_test.txt \
--freq_modes 8 \
--time_subsample 1 \
--epochs 1000
