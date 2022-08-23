#!/bin/bash

module load julia/1.7.3

GemmDenseThreadsDIR=../../simple-gemm/julia/GemmDenseThreads
# Modify this file for double, float or float16
EXECUTABLE=$GemmDenseThreadsDIR/gemm-dense-threads.jl
export JULIA_EXCLUSIVE=1

Ms=( 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 )
t=80
REPETITIONS=5

for M in "${Ms[@]}"; do

    salloc -N 1 -p Ampere -t 10:00:00 srun -n 1 -c $t \
      julia -O3 --project=$GemmDenseThreadsDIR -t $t $EXECUTABLE $M $M $M $REPETITIONS > Ampere-Julia173-${t}t-${M}M_${REPETITIONS}s_F16_ex.log 2>&1 &
done

