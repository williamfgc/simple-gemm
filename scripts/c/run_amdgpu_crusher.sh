#!/bin/bash
#SBATCH -A CSC383_crusher
#SBATCH -J M_C_gpu_Float32
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1


PROJDIR=../../simple-gemm/c
EXECUTABLE=$PROJDIR/gemm-dense-hip

module load rocm/5.2.0 cray-mpich

Ms=( 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 )

for M in "${Ms[@]}"; do

  start_time=$(date +%s)
  srun -n 1 --gpus=1 $EXECUTABLE $M $M $M 5
  end_time=$(date +%s)

  # elapsed time with second resolution
  elapsed=$(( end_time - start_time ))
  echo simple-gemm language=c compiler=hipcc_5_2_0 size=$M gpu time=$elapsed "seconds"

done
