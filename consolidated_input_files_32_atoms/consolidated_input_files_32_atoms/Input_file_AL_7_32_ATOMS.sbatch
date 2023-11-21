#!/bin/bash
#SBATCH -J AL_51_TRICLINIC_4_ATOMS_FOR_STRUCTURAL_RELAXATION
#SBATCH -A gts-phanish6
#SBATCH -q embers
#SBATCH -N 28 --ntasks-per-node=24
#SBATCH --mem=0
#SBATCH -oReport-%j.out
#SBATCH -t1:00:00
cd $SLURM_SUBMIT_DIR

module reset
module load mkl
echo $PWD

srun ../lib/sparc -name AL_7_32_ATOMS
