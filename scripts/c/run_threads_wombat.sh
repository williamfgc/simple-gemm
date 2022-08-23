#!/bin/bash

module load ARM_Compiler_For_HPC/22.0

# Modify this file for double, float or float16
EXECUTABLE=../../simple-gemm/c/gemm-dense-openmp-armclang

Ms=( 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 )
t=80
REPETITIONS=5

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=$t

for M in "${Ms[@]}"; do

  salloc -N 1 -p Ampere -t 10:00:00 srun -n 1 -c $t \
      $EXECUTABLE $M $M $M $REPETITIONS > Ampere-ARMClang22-${t}t-${M}M_${REPETITIONS}s_spread_threads.log 2>&1 &
done

