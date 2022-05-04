#!/bin/bash

EXECUTABLE=./gemm-dense-thread-g++
M=10000
threads=(1 2 4 8 12 16 20 22 24)

for t in "${threads[@]}"
do
  echo "Running with $t threads for M=$M"
  export OMP_NUM_THREADS=$t
  time $EXECUTABLE $M $M $M > M10K_${t}t.log
done


