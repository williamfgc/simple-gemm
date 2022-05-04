#!/bin/bash

EXECUTABLE=../../fortran/gemm-dense-openmp-armflang

M=16384
threads=( 80 70 60 50 40 30 20 10 5 2 1 )

for t in "${threads[@]}"; do
    export NUMBA_NUM_THREADS=$t
    salloc -N 1 -p Ampere -t 10:00:00 srun -n 1 -c $t \
    $EXECUTABLE $M $M $M > Ampere-ARMFlang22-${M}M-${t}t.log 2>&1
done
