#!/bin/bash
#SBATCH -J AL_32_atoms
#SBATCH -A gts-phanish6
#SBATCH -q embers
#SBATCH -N 38 --ntasks-per-node=24
#SBATCH --mem=0
#SBATCH -oReport-%j.out
#SBATCH -t1:00:00
cd $SLURM_SUBMIT_DIR

module reset
module load mkl
echo $PWD

srun ../lib/sparc -name AL_108_ATOMS_SET_3
