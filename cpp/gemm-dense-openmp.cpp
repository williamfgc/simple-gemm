
// From
// https://repository.prace-ri.eu/git/CodeVault/hpc-kernels/dense_linear_algebra/-/blob/master/gemm/gemm_openmp/src/gemm_openmp.cpp
// Apache v2 license

#include <chrono>
#include <iostream>
#include <random>

#include <omp.h>

namespace {
void fill_random(float *A, const int &n, const int &m) {
  std::mt19937 e(std::chrono::duration_cast<std::chrono::hours>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count());
  std::uniform_real_distribution<float> f;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      A[i * m + j] = f(e);
    }
  }
}

void gemm(float *A, float *B, float *C, const int &A_rows, const int &A_cols,
          const int &B_rows) {

  int i, k, j;
#pragma omp parallel for shared(A, B, C, A_rows, A_cols, B_rows) private(i, k, \
                                                                         j)
  for (i = 0; i < A_rows; i++) {
    for (k = 0; k < A_cols; k++) {
      float sum = 0.0f;
      for (j = 0; j < B_rows; j++) {
        sum += A[i * A_cols + k] * B[k * B_rows + j];
      }
      C[i * B_rows + j] = sum;
    }
  }
}

} // namespace

int main(int argc, char *argv[]) {

  int A_rows, A_cols, B_rows, B_cols;

  if (argc != 4) {
    std::cout
        << "Usage: 3 arguments: matrix A rows, matrix A cols and matrix B cols"
        << std::endl;
    return 1;
  } else {
    A_rows = atoi(argv[1]);
    A_cols = atoi(argv[2]);
    B_rows = atoi(argv[2]);
    B_cols = atoi(argv[3]);
  }

  float *A = new float[A_rows * A_cols];
  float *B = new float[B_rows * B_cols];
  float *C = new float[A_rows * B_cols](); // value-init to zero

  fill_random(A, A_rows, A_cols);
  fill_random(B, B_rows, B_cols);

  const auto t_start = std::chrono::high_resolution_clock::now();
  gemm(A, B, C, A_rows, A_cols, B_cols);
  const auto t_end = std::chrono::high_resolution_clock::now();
  const double dtime =
      std::chrono::duration<double, std::milli>(t_end - t_start).count() / 1E6;

  const double total_memory_GB = (A_rows * A_cols) * (B_rows * B_cols) *
                                 (A_rows * B_cols) * sizeof(float) / 1E9;

  std::cout << "ordinary gemm time: " << dtime << " s for " << A_rows << " "
            << B_cols << " rows columns. Memory = " << total_memory_GB << " GB"
            << std::endl;

  delete[] A;
  delete[] B;
  delete[] C;
  return 0;
}
