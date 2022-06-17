#!/bin/bash

#SBATCH --job-name=test
#SBATCH --qos=condo
#SBATCH --partition=shas
#SBATCH --account=ucb-summit-smr
#SBATCH --output=sim.log
#SBATCH --error=sim.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2

export OMP_NUM_THREADS=2
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

srun -n1 --mpi=pmi2 aLENS.X
