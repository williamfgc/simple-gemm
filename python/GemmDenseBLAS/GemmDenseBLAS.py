
import sys
import typing
import numpy as np
import time
from scipy.linalg import blas


def gemm(A: np.ndarray, B: np.ndarray, C: np.ndarray):

    blas.sgemm(1., A, B, 0, C)
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

    args = sys.argv[1:]
    print(args)

    # args don't include Julia executable and program
    if len(args) != 3:
        raise ValueError(
            "Usage: 3 arguments: matrix A rows, matrix A cols and matrix B cols")
    else:
        A_rows = int(args[0])
        A_cols = int(args[1])
        B_rows = int(args[1])
        B_cols = int(args[2])

    rng = np.random.default_rng()
    start = time.time()
    A = rng.random((A_rows, A_cols), dtype=np.float32)
    tmp = _print_time(start, "initialize A")
    B = rng.random((B_rows, B_cols), dtype=np.float32)
    tmp = _print_time(tmp, "initialize B")
    C = np.zeros(dtype=np.float32, shape=(A_rows, B_cols))
    tmp = _print_time(tmp, "initialize C")

    gemm(A, B, C)
    tmp = _print_time(tmp, "simple gemm")
    tmp = _print_time(start, "total time")

    # print(C)

    return 0


if __name__ == "__main__":
    main()
