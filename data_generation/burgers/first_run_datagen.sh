#!/bin/bash

#SBATCH --job-name=gen_data_N_256_s_512
#SBATCH --partition=debug
#SBATCH --output=/home/meliao/projects/fourier_neural_operator/data_generation/burgers/00_gen_data.out
#SBATCH --error=/home/meliao/projects/fourier_neural_operator/data_generation/burgers/00_gen_data.err
echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"

if [[ ! -d ~/projects/fourier_neural_operator/data/2021-05-10_Burgers_multiple_time_values ]]; then
  mkdir -p ~/projects/fourier_neural_operator/data/2021-05-10_Burgers_multiple_time_values
fi

matlab \
 -batch "N=4;seed=0;s=1024;tmax=5;n_tsteps=200;out_fp='~/projects/fourier_neural_operator/data/2021-05-10_Burgers_multiple_time_values/test_out.mat';gen_burgers_multiple_time_values;"
