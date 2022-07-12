#!/bin/bash

module load nvhpc-nompi/22.1
export JULIA_CUDA_USE_BINARYBUILDER=false

GemmDenseCUDADIR=../../simple-gemm/julia/GemmDenseCUDA
EXECUTABLE=$GemmDenseCUDADIR/gemm-dense-cuda.jl

# maximum theoretical = 206559
#M=16384
M=32768
REPETITIONS=10

salloc -N 1 -p Ampere -t 10:00:00 --gres=gpu:1 srun -n 1 \
    julia --project=$GemmDenseCUDADIR \
    $EXECUTABLE $M $M $M $REPETITIONS > A100-Julia1_7_3-${M}M_${steps}s.log 2>&1
