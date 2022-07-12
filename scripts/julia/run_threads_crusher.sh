#!/bin/bash
#SBATCH -A CSC383_crusher
#SBATCH -J 65536M_julia_32_16
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 8:00:00
#SBATCH -p batch
#SBATCH -N 1

#M=206559 # max possible matrix size for 512 GB
#M=206559 # max possible matrix size for single NUMA region 128 GB

#M=16384 # 2^14
#M=32768 # 2^15
M=65536 # 2^16
#M=131072 # 2^17

# Modify as required
PROJDIR=../../julia/GemmDenseThreads
EXECUTABLE=$PROJDIR/gemm-dense-threads.jl

#threads=( 128 108 84 64 32 16 8 4 2 1 )
# typicallly to cover 2 or 1 NUMA regions
threads=( 32 16 )


for t in "${threads[@]}"; do

  start_time=$(date +%s)
  srun -n 1 --ntasks-per-node=1 -c $t julia -O3 -t $t --project=$PROJDIR $EXECUTABLE $M $M $M
  end_time=$(date +%s)

  # elapsed time with second resolution
  elapsed=$(( end_time - start_time ))
  echo simple-gemm language=julia compiler=1.8.0-rc1 size=$M threads=$t time=$elapsed "seconds"
done
