#!/bin/bash

GemmDenseThreadsDIR=../../julia/GemmDenseThreads
EXECUTABLE=$GemmDenseThreadsDIR/gemm-dense-threads.jl

# maximum theoretical = 206559
M=16384
threads=( 80 70 60 50 40 30 20 10 5 2 1 )

for t in "${threads[@]}"; do
    salloc -N 1 -p Ampere -t 10:00:00 srun -n 1 -c $t \
    julia -t $t --project=$GemmDenseThreadsDIR $EXECUTABLE $M $M $M > Ampere-Julia1_7_2-${M}M-${t}t.log 2>&1
done
