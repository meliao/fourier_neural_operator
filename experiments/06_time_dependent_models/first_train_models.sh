#!/bin/bash

#SBATCH --job-name=00_first_job
#SBATCH --time=4:00:00
#SBATCH --partition=contrib-gpu
#SBATCH --output=experiments/06_time_dependent_models/logs/00_first_job.out
#SBATCH --error=experiments/06_time_dependent_models/logs/00_first_job.err

echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"

source  ~/conda_init.sh
cd ~/projects/fourier_neural_operator/
conda activate fourier_neural_operator

python -m experiments.06_time_dependent_models.train_models \
--data_fp data/2021-06-05_NLS_ten_step_testing_dataset.mat \
--model_fp experiments/06_time_dependent_models/models/first_model \
--results_fp experiments/06_time_dependent_models/results/first_results.txt \
--train_df experiments/06_time_dependent_models/results/first_model.txt \
--freq_modes 8 \
--max_tsteps 4