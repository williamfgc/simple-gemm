#!/bin/bash
#SBATCH -A CSC383_crusher
#SBATCH -J M_python_cpu_F32_64t
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 3:00:00
#SBATCH -p batch
#SBATCH -N 1

PROJDIR=../../simple-gemm/python/GemmDenseThreads
# Modify this for double, float or float16
EXECUTABLE=$PROJDIR/GemmDenseThreads.py

module load cray-python/3.9.12.1 cray-mpich

Ms=( 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 )
threads=64
export NUMBA_NUM_THREADS=$threads

for M in "${Ms[@]}"; do

  start_time=$(date +%s)
  srun -n 1 -c $threads python3 $EXECUTABLE $M $M $M 5
  end_time=$(date +%s)

  # elapsed time with second resolution
  elapsed=$(( end_time - start_time ))
  echo simple-gemm language=python compiler=3.9.12.1 size=$M cpu time=$elapsed "seconds"

done
