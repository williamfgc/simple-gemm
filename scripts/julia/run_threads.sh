#!/bin/bash

GemmDenseThreadsDIR=../../julia/GemmDenseThreads
EXECUTABLE=gemm-dense-threads.jl
M=10000
threads=(1 2 4 8 12 16 20 22 24)

for t in "${threads[@]}"
do
  echo "Running with $t threads for M=$M"
  time julia -t $t --project=$GemmDenseDIR $EXECUTABLE $M $M $M > M10K_${t}t.log
done


