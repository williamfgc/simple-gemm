
// From
// https://repository.prace-ri.eu/git/CodeVault/hpc-kernels/dense_linear_algebra/-/blob/master/gemm/gemm_openmp/src/gemm_openmp.cpp
// Apache v2 license

#include <chrono>
#include <iostream>
#include <random>

#include "cblas.h"

namespace {
void fill_random(float *A, const int n, const int m, std::mt19937 &r) {

  std::uniform_real_distribution<float> f(0, 1);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      A[i * m + j] = f(r);
    }
  }
}

std::chrono::system_clock::time_point
print_dtime(const std::chrono::system_clock::time_point &start,
            const std::string &hint) {

  const auto end = std::chrono::high_resolution_clock::now();
  const double dtime =
      std::chrono::duration<double, std::milli>(end - start).count() / 1E3;
  std::cout << "Time to " << hint << " " << dtime << " s\n";
  return end;
}

} // namespace

int main(int argc, char *argv[]) {

  // Set seed once
  std::mt19937 e(std::chrono::duration_cast<std::chrono::hours>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count());

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

  const auto start = std::chrono::high_resolution_clock::now();
  float *A = new float[A_rows * A_cols];
  auto tmp = print_dtime(start, "allocate A");
  float *B = new float[B_rows * B_cols];
  tmp = print_dtime(tmp, "allocate B");
  float *C = new float[A_rows * B_cols](); // value-init to zero
  tmp = print_dtime(tmp, "initialize C");

  fill_random(A, A_rows, A_cols, e);
  tmp = print_dtime(tmp, "fill A");
  fill_random(B, B_rows, B_cols, e);
  tmp = print_dtime(tmp, "fill B");

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              static_cast<blasint>(A_rows), static_cast<blasint>(B_cols),
              static_cast<blasint>(A_cols), 1.0f, A,
              static_cast<blasint>(A_cols), B, static_cast<blasint>(B_rows), 0,
              C, static_cast<blasint>(B_cols));

  tmp = print_dtime(tmp, "cblas gemm");
  tmp = print_dtime(start, "total time");

  delete[] A;
  tmp = print_dtime(tmp, "free A");
  delete[] B;
  tmp = print_dtime(tmp, "free B");
  delete[] C;
  tmp = print_dtime(tmp, "free C");
  return 0;
}
