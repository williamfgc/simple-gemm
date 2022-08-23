!/bin/bash
#SBATCH -A CSC383_crusher
#SBATCH -J M_julia_cpu_F16_64t_ex
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 6:00:00
#SBATCH -p batch
#SBATCH -N 1

PROJDIR=../../simple-gemm/julia/GemmDenseThreads
# this file needs to be modified to target double, float or float16 cases
EXECUTABLE=$PROJDIR/gemm-dense-threads.jl

module load cray-mpich
export JULIA_MPIEXEC=srun
# Thread policy to bind threads
export JULIA_EXCLUSIVE=1

Ms=( 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 )
threads=64

for M in "${Ms[@]}"; do

  start_time=$(date +%s)
  srun -n 1 -c $threads julia -O3 --project=$PROJDIR -t $threads $EXECUTABLE $M $M $M 5
  end_time=$(date +%s)

  # elapsed time with second resolution
  elapsed=$(( end_time - start_time ))
  echo simple-gemm language=julia compiler=1.8.0-rc1 size=$M cpu time=$elapsed "seconds"

done
