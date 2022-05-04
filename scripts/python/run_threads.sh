#!/bin/bash

EXECUTABLE=../../python/GemmDenseThreads.py
M=10000
threads=(1 2 4 8 12 16 20 22 24)

for t in "${threads[@]}"
do
  echo "Running with $t threads for M=$M"
  export NUMBA_NUM_THREADS=$t
  time python3 -t $t --project=$GemmDenseDIR $EXECUTABLE $M $M $M > M10K_${t}t.log
done


