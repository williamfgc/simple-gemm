#!/bin/bash

module load python/3.9.9

GemmDenseThreadsDIR=../../simple-gemm/python/GemmDenseThreads
# Modify this file for double, float or float16
EXECUTABLE=$GemmDenseThreadsDIR/GemmDenseThreads.py

Ms=( 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 )
threads=( 80 )
REPETITIONS=5

for t in "${threads[@]}"; do
  export NUMBA_NUM_THREADS=$t
# These didn't make any difference
# export NUMBA_THREADING_LAYER=omp
# export OMP_PROC_BIND=true
# export OMP_NUM_THREADS=$threads

  for M in "${Ms[@]}"; do

    salloc -N 1 -p Ampere -t 10:00:00 srun -n 1 -c $t \
      python3 $EXECUTABLE $M $M $M $REPETITIONS > Ampere-Python399-${t}t-${M}M_${REPETITIONS}s_F16.log 2>&1 &
  done
done
