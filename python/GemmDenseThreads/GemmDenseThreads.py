
import sys
import typing
from numba import njit, prange
import numpy as np


@njit(parallel=True, nogil=True, fastmath=True)
def gemm(A: np.ndarray, B: np.ndarray, C: np.ndarray):

    A_rows = A.shape[0]
    A_cols = A.shape[1]
    B_cols = B.shape[1]

    # here set the parallel range "prange"
    for i in prange(0, A_rows):
        for k in range(0, A_cols):
            temp = A[i, k]
            for j in range(0, B_cols):
                C[i, j] += temp * B[k, j]

    return


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
    A = rng.random((A_rows, A_cols), dtype=np.float32)
    B = rng.random((B_rows, B_cols), dtype=np.float32)
    C = np.zeros(dtype=np.float32, shape=(A_rows, B_cols))

    gemm(A, B, C)

    # print(C)

    return 0


if __name__ == "__main__":
    main()
