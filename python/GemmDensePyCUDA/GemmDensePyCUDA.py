
import sys
import typing
from pycuda import driver, compiler, gpuarray, tools
import numpy as np
import time
from math import ceil

BLOCK_SIZE = 32

kernel_gemm = """
__global__ void gemm(float *A, float *B, float *C, int64_t A_rows,
                     int64_t A_cols, int64_t B_cols) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  int i;

  if (col < B_cols && row < A_rows) {
    for (i = 0; i < A_cols; ++i) {
      sum += A[row * A_cols + i] * B[i * B_cols + col];
    }
    C[row * B_cols + col] = sum;
  }
}
"""


def _print_time(start, process: str):
    end = time.time()
    print("Time to " + process + " : " + str(end-start) + " s")
    return end


def main():

    # must initialize scalars
    A_rows: int = -1
    A_cols: int = -1
    B_rows: int = -1
    B_cols: int = -1
    steps: int = 1

    args = sys.argv[1:]
    print(args)

    # args don't include the python executable and program
    argc = len(args)

    if argc == 3 or argc == 4:
        A_rows = int(args[0])
        A_cols = int(args[1])
        B_rows = int(args[1])
        B_cols = int(args[2])
        if argc == 4:
            steps = int(args[3])
    else:
        raise ValueError(
            "Usage: 3 arguments: matrix A rows, matrix A cols and matrix B cols\n"
            "Usage: 4 arguments: matrix A rows, matrix A cols and matrix B cols and steps\n")

    rng = np.random.default_rng()
    start = time.time()
    A = rng.random((A_rows, A_cols), dtype=np.float32)
    tmp = _print_time(start, "initialize A")
    B = rng.random((B_rows, B_cols), dtype=np.float32)
    tmp = _print_time(tmp, "initialize B")
    # C = np.zeros(dtype=np.float32, shape=(A_rows, B_cols))
    # tmp = _print_time(tmp, "initialize C")

    A_d = gpuarray.to_gpu(A)
    tmp = _print_time(tmp, "copy A")
    B_d = gpuarray.to_gpu(B)
    tmp = _print_time(tmp, "copy B")
    C_d = gpuarray.empty((A_rows, B_cols), np.float32)
    tmp = _print_time(tmp, "initialize C")

    grid_rows = ceil((A_rows + BLOCK_SIZE - 1) / BLOCK_SIZE)
    grid_cols = ceil((B_cols + BLOCK_SIZE - 1) / BLOCK_SIZE)
    blocks = (grid_rows, grid_cols)
    threads = (BLOCK_SIZE, BLOCK_SIZE)

    # compile the kernel code
    mod = compiler.SourceModule(kernel_gemm)
    gemm = mod.get_function("gemm")

    # call the kernel
    gemm(A_d, B_d, C_d, A_rows, B_rows, B_cols, block=blocks, threads=threads)
    driver.Context.synchronize()
    tmp = _print_time(tmp, "simple gemm")

    if steps > 1:

        average_time = 0.
        for i in range(1, steps):
            start = time.time()
            gemm(A_d, B_d, C_d, A_rows, B_rows, B_cols)
            driver.Context.synchronize()
            end = time.time()
            print("Time to simple gemm : " + str(end-start) + " s")
            average_time += (end-start)

        average_time /= steps-1
        gflops = (2 * A_rows * A_cols * B_cols*1E-9)/average_time

        print("GFLOPS: " + str(gflops) + " steps: " + str(steps) + "\n")

    tmp = _print_time(start, "total time")

    # print(A)
    # print(B)
    # print(C)

    return 0


if __name__ == "__main__":
    main()
