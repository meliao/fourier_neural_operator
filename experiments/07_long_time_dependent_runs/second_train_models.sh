#!/bin/bash

#SBATCH --job-name=02_time_dep
#SBATCH --partition=contrib-gpu-long
#SBATCH --output=experiments/07_long_time_dependent_runs/logs/02_time_dep.out
#SBATCH --error=experiments/07_long_time_dependent_runs/logs/02_time_dep.err
#SBATCH --exclude=gpu-g16,gpu-g28,gpu-g29,gpu-g38


echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"

source  ~/conda_init.sh
cd ~/projects/fourier_neural_operator/
conda activate fourier_neural_operator

python -m experiments.07_long_time_dependent_runs.train_models \
--data_fp /share/data/willett-group/meliao/data/2021-06-24_NLS_data_04_train.mat \
--test_data_fp /share/data/willett-group/meliao/data/2021-06-24_NLS_data_04_test.mat \
--model_fp experiments/07_long_time_dependent_runs/models/02_time_dep_ep_{} \
--train_df experiments/07_long_time_dependent_runs/results/02_time_dep_train.txt \
--test_df experiments/07_long_time_dependent_runs/results/02_time_dep_test.txt \
--freq_modes 8 \
--time_subsample 1 \
--epochs 5000
