#!/bin/bash
#SBATCH -A CSC383_crusher
#SBATCH -J M_C_cpu_F32_64t_true_threads
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 3:00:00
#SBATCH -p batch
#SBATCH -N 1

# modify with absolute path if different to point at where executables are built in simple-gemm
PROJDIR=../../simple-gemm/c
# here put the corresponding executable
# float: gemm-dense-openmp or double: gemm-dense-openmp64 
EXECUTABLE=$PROJDIR/gemm-dense-openmp

module load rocm/5.2.0 cray-mpich

Ms=( 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 )
threads=64
REPETITIONS=5

export OMP_PROC_BIND=true
export OMP_NUM_THREADS=$threads
export OMP_PLACES=threads

for M in "${Ms[@]}"; do

  start_time=$(date +%s)
  srun -n 1 -c $threads $EXECUTABLE $M $M $M $REPETITIONS
  end_time=$(date +%s)

  # elapsed time with second resolution
  elapsed=$(( end_time - start_time ))
  echo simple-gemm language=c compiler=amdclang-5.2.0 size=$M cpu time=$elapsed "seconds"

done
