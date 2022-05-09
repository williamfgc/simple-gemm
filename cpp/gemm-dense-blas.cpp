
// From
// https://repository.prace-ri.eu/git/CodeVault/hpc-kernels/dense_linear_algebra/-/blob/master/gemm/gemm_openmp/src/gemm_openmp.cpp
// Apache v2 license

#include <gemm-dense-common.h>

#include <iostream>

#include "cblas.h"

namespace {

void gemm(float *A, float *B, float *C, const int A_rows, const int A_cols,
          const int B_rows, const int B_cols) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A_rows, B_cols,
                A_cols, 1.0f, A, A_cols, B, B_rows, 0, C, B_cols);
}

}  // namespace

int main(int argc, char *argv[]) {
    // Set seed once
    std::mt19937 e(std::chrono::duration_cast<std::chrono::hours>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count());

    int A_rows, A_cols, B_rows, B_cols;

    if (argc != 4) {
        std::cout << "Usage: 3 arguments: matrix A rows, matrix A cols and "
                     "matrix B cols"
                  << std::endl;
        return 1;
    } else {
        A_rows = atoi(argv[1]);
        A_cols = atoi(argv[2]);
        B_rows = atoi(argv[2]);
        B_cols = atoi(argv[3]);
    }

    const auto start = std::chrono::high_resolution_clock::now();
    float *A = new float[A_rows * A_cols];
    auto tmp = gd::print_dtime(start, "allocate A");
    float *B = new float[B_rows * B_cols];
    tmp = gd::print_dtime(tmp, "allocate B");
    float *C = new float[A_rows * B_cols]();  // value-init to zero
    tmp = gd::print_dtime(tmp, "initialize C");

    gd::fill_random(A, A_rows, A_cols, e);
    tmp = gd::print_dtime(tmp, "fill A");
    gd::fill_random(B, B_rows, B_cols, e);
    tmp = gd::print_dtime(tmp, "fill B");

    gemm(A, B, C, A_rows, A_cols, B_rows, B_cols);
    tmp = gd::print_dtime(tmp, "cblas gemm");
    tmp = gd::print_dtime(start, "total time");

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
