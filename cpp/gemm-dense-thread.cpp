
// From
// https://repository.prace-ri.eu/git/CodeVault/hpc-kernels/dense_linear_algebra/-/blob/master/gemm/gemm_openmp/src/gemm_openmp.cpp
// Apache v2 license

#include <algorithm> //std::for_each
#include <chrono>
#include <execution> //std::algorithm
#include <iostream>
#include <numeric> //std::iota
#include <random>
#include <thread>

namespace {
void fill_random(float *A, const int n, const int m, std::mt19937 &r) {

  std::uniform_real_distribution<float> f(0, 1);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      A[i * m + j] = f(r);
    }
  }
}

void gemm(float *A, float *B, float *C, const int A_rows, const int A_cols,
          const int B_rows) {

  auto lf_gemm_thread = [](float *A, float *B, float *C, const int A_cols,
                           const int B_rows, const int start,
                           const int stride) {
    for (int i = start; i < (start + stride); ++i) {
      for (int k = 0; k < A_cols; ++k) {
        float temp = A[i * A_cols + k];
        for (int j = 0; j < B_rows; ++j) {
          C[i * B_rows + j] += temp * B[k * B_rows + j];
        }
      }
    }
  };

  const int nthreads = std::stoi(std::getenv("OMP_NUM_THREADS"));

  std::vector<std::thread> threads;
  threads.reserve(nthreads);

  const int nindices = A_rows / nthreads;

  for (int tid = 0; tid < nthreads; ++tid) {
    const int start = nindices * tid;
    const int stride =
        (tid == nthreads - 1) ? nindices + A_rows % nthreads : nindices;

    threads.push_back(
        std::thread(lf_gemm_thread, A, B, C, A_cols, B_rows, start, stride));
  }

  for (auto &t : threads) {
    t.join();
  }
}

std::chrono::system_clock::time_point
print_dtime(const std::chrono::system_clock::time_point &start,
            const std::string &hint) {

  const auto end = std::chrono::high_resolution_clock::now();
  const double dtime =
      std::chrono::duration<double, std::milli>(end - start).count() / 1E6;
  std::cout << "Time to " << hint << " " << dtime << " s\n";
  return end;
}

static void print_matrix(float *A, const int A_rows, const int A_cols) {

  std::cout << "[ ";
  for (int i = 0; i < A_rows; ++i) {
    for (int j = 0; j < A_cols; ++j) {
      std::cout << A[i * A_cols + j] << " ";
    }
    std::cout << ";\n";
  }
  std::cout << "]\n";
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
  gemm(A, B, C, A_rows, A_cols, B_cols);
  tmp = print_dtime(tmp, "simple gemm");
  tmp = print_dtime(start, "total time");

  print_matrix(A, A_rows, A_cols);
  print_matrix(B, B_rows, B_cols);
  print_matrix(C, A_rows, B_cols);

  delete[] A;
  tmp = print_dtime(tmp, "free A");
  delete[] B;
  tmp = print_dtime(tmp, "free B");
  delete[] C;
  tmp = print_dtime(tmp, "free C");
  return 0;
}
