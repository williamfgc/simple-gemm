
import sys
import typing
from numba import njit, prange
import numpy as np
import time


@njit(parallel=True, nogil=True)
def gemm(A: np.ndarray, B: np.ndarray, C: np.ndarray):

    A_rows = A.shape[0]
    A_cols = A.shape[1]
    B_cols = B.shape[1]

    # here set the parallel range "prange"
    for i in prange(0, A_rows):
        # for i in range(0, A_rows):
        for k in range(0, A_cols):
            temp = A[i, k]
            for j in range(0, B_cols):
                C[i, j] += temp * B[k, j]

    return


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
    A = rng.random((A_rows, A_cols), dtype=np.float64)
    tmp = _print_time(start, "initialize A")
    B = rng.random((B_rows, B_cols), dtype=np.float64)
    tmp = _print_time(tmp, "initialize B")
    C = np.zeros(dtype=np.float64, shape=(A_rows, B_cols))
    tmp = _print_time(tmp, "initialize C")

    gemm(A, B, C)
    tmp = _print_time(tmp, "simple gemm")

    if steps > 1:

        average_time = 0.
        for i in range(1, steps):
            start = time.time()
            gemm(A, B, C)
            end = time.time()
            print("Time to simple gemm : " + str(end-start) + " s")
            average_time += (end-start)

        average_time /= steps-1
        gflops = (2 * A_rows * A_cols * B_cols*1E-9)/average_time

        print("GFLOPS: " + str(gflops) + " steps: " + str(steps) +
              " average_time: " + str(average_time) + "\n")

    tmp = _print_time(start, "total time")

    # print(C)

    return 0


if __name__ == "__main__":
    main()
