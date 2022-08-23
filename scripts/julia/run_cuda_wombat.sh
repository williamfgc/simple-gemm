#!/bin/bash

module load nvhpc-nompi/22.1
module load julia/1.7.3 
export JULIA_CUDA_USE_BINARYBUILDER=false

GemmDenseCUDADIR=../../simple-gemm/julia/GemmDenseCUDA
# Modify this file for double, float, float16
EXECUTABLE=$GemmDenseCUDADIR/gemm-dense-cuda.jl
REPETITIONS=5

Ms=( 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 )

for M in "${Ms[@]}"; do

  salloc -N 1 -p Ampere -t 10:00:00 --gres=gpu:1 srun -n 1 \
    julia -O3 --project=$GemmDenseCUDADIR \
    $EXECUTABLE $M $M $M $REPETITIONS > A100-Julia1_7_3-${M}M_${REPETITIONS}s_F32_simd.log 2>&1 &
done
