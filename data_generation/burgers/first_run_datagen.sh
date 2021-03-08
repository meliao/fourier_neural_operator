#!/bin/bash

#SBATCH --job-name=gen_data_N_256_s_512
#SBATCH --partition=debug
#SBATCH --output=/home/meliao/projects/fourier_neural_operator/data_generation/burgers/00_gen_data.out
#SBATCH --error=/home/meliao/projects/fourier_neural_operator/data_generation/burgers/00_gen_data.err
echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"

matlab \
 -batch "N=10;seed=0;s=1024;out_fp='/home/meliao/projects/fourier_neural_operator/data/00_Burgers_seed_0.mat';gen_burgers1;"
