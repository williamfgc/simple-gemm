
import sys
import typing
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
import numpy as np
import time
from math import ceil

BLOCK_SIZE = 32


@cuda.jit
def gemm(A: DeviceNDArray, B: DeviceNDArray, C: DeviceNDArray):

    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


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
    C = np.zeros(dtype=np.float32, shape=(A_rows, B_cols))
    tmp = _print_time(tmp, "initialize C")

    grid_rows = ceil((A_rows + BLOCK_SIZE - 1) / BLOCK_SIZE)
    grid_cols = ceil((B_cols + BLOCK_SIZE - 1) / BLOCK_SIZE)
    blocks = (grid_rows, grid_cols)
    threads = (BLOCK_SIZE, BLOCK_SIZE)

    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)
    gemm[blocks, threads](d_A, d_B, d_C)
    cuda.synchronize()
    tmp = _print_time(tmp, "simple gemm")

    if steps > 1:

        average_time = 0.
        for i in range(1, steps):
            start = time.time()
            gemm[blocks, threads](d_A, d_B, d_C)
            cuda.synchronize()
            end = time.time()
            print("Time to simple gemm : " + str(end-start) + " s")
            average_time += (end-start)

        average_time /= steps-1
        gflops = (2 * A_rows * A_cols * B_cols*1E-9)/average_time

        print("GFLOPS: " + str(gflops) + " steps: " + str(steps) +
              " average_time: " + str(average_time) + "\n")

    tmp = _print_time(start, "total time")

    # print(A)
    # print(B)
    # print(C)

    return 0


if __name__ == "__main__":
    main()
