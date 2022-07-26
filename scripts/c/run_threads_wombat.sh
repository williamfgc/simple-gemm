#!/bin/bash

# assummes runs are separated from the simple-gemm source code
EXECUTABLE=../../simple-gemm/c/gemm-dense-openmp-armclang
# Float64: EXECUTABLE=../../simple-gemm/c/gemm-dense-openmp-armclang64

module load ARM_Compiler_For_HPC/22.0

Ms=( 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 )

threads=( 80 )

export OMP_PROC_BIND=spread
export OMP_PLACES=threads

for t in "${threads[@]}"; do
  export OMP_NUM_THREADS=$t	
  for M in "${Ms[@]}"; do

    salloc -N 1 -p Ampere -t 10:00:00 srun -n 1 -c $t \
      $EXECUTABLE $M $M $M 5 > Ampere-ARMClang22-${t}t-${M}M_5s_F32_spread_threads.log 2>&1 &
      # $EXECUTABLE $M $M $M 5 > Ampere-ARMClang22-${t}t-${M}M_5s_F64_spread_threads.log 2>&1 &
  done
done
