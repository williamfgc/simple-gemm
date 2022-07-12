#!/bin/bash
#SBATCH -A CSC383_crusher
#SBATCH -J 60000M_julia_gpu
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 0:02:00
#SBATCH -p batch
#SBATCH -N 1

#N=206559 # max possible matrix size for 512 GB
#M=16384
#M=32768
#=65536
M=60000
#M=8192
#M=131072
REPETITIONS=5

PROJDIR=../../simple-gemm/julia/GemmDenseAMDGPU
EXECUTABLE=$PROJDIR/gemm-dense-amdgpu.jl

module load rocm cray-mpich
# rocminfo | grep GPU optional
export JULIA_MPIEXEC=srun
# required to make sure AMDGPU.jl uses system's rocm
export JULIA_AMDGPU_DISABLE_ARTIFACTS=1 
echo "ROCM_VISIBLE_DEVICES= " $ROCR_VISIBLE_DEVICES

start_time=$(date +%s)
srun -n 1 --gpus=1 julia --project=$PROJDIR $EXECUTABLE $M $M $M $REPETITIONS
end_time=$(date +%s)

# elapsed time with second resolution
elapsed=$(( end_time - start_time ))
echo simple-gemm language=julia compiler=1.8.0-rc1 size=$M gpu time=$elapsed "seconds"
