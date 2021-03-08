#!/bin/bash

#SBATCH --job-name=gen_data_N_256_s_512
#SBATCH --partition=general
#SBATCH --output=/home/meliao/projects/fourier_neural_operator/data_generation/burgers/gen_data_N_256_s_512.out
#SBATCH --error=/home/meliao/projects/fourier_neural_operator/data_generation/burgers/gen_data_N_256_s_512.err
echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"

matlab \
 -batch \
"N=100;
seed=0;
s=8192;
out_fp=~/projects/fourier_neural_operator/data/01_test_Burgers.mat;
gen_burgers1;"
