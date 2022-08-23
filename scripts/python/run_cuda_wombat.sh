#!/bin/bash

module load python/3.9.9 
module load nvhpc-nompi/22.1

GemmDenseCUDADIR=../../simple-gemm/python/GemmDenseCUDA
# Modify this file for double, float or float16
EXECUTABLE=$GemmDenseCUDADIR/GemmDenseCUDA.py

REPETITIONS=5

export NUMBA_CUDA_USE_NVIDIA_BINDING=1

Ms=( 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 )

for M in "${Ms[@]}"; do

  salloc -N 1 -p Ampere -t 10:00:00 --gres=gpu:1 srun -n 1 \
    python3 \
    $EXECUTABLE $M $M $M $REPETITIONS > A100-Python3_9_9-${M}M_${REPETITIONS}s_float16.log 2>&1 &
done
