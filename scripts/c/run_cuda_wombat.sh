#!/bin/bash

module load nvhpc-nompi/22.1
# this file needs to be modified to target double, float or float16 cases
EXECUTABLE=../../simple-gemm/c/gemm-dense-cuda

Ms=( 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 )

for M in "${Ms[@]}"; do

  salloc -N 1 -p Ampere -t 10:00:00 --gres=gpu:1 srun -n 1 \
    $EXECUTABLE $M $M $M 5 > A100-cuda-nvhpc22_1-${M}M_5s_float32.log 2>&1 &
done
