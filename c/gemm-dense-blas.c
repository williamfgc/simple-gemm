#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cblas.h"

static void fill_random(double *A, const int64_t n, const int64_t m) {

  int64_t i, j;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < m; ++j) {
      A[i * m + j] = (double)rand() / (double)RAND_MAX;
    }
  }
}

static void gemm(double *A, double *B, double *C, const int64_t A_rows,
                 const int64_t A_cols, const int64_t B_cols) {
    
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A_rows, B_cols,
                A_cols, 1.0f, A, A_cols, B, B_rows, 0, C, B_cols);
}

static struct timespec print_dtime(struct timespec start, const char *process) {
  struct timespec end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  const double dtime = ((double)((end.tv_sec - start.tv_sec) * 1000000 +
                                 (end.tv_nsec - start.tv_nsec) / 1000)) /
                       1E6;

  printf("Time to %s = %f s\n", process, dtime);
  return end;
}

static double dtime(struct timespec start, struct timespec end) {
  return ((double)((end.tv_sec - start.tv_sec) * 1000000 +
                   (end.tv_nsec - start.tv_nsec) / 1000)) /
         1E6;
}

static void print_matrix(double *A, const int64_t A_rows,
                         const int64_t A_cols) {

  int64_t i, j;
  printf("[");
  for (i = 0; i < A_rows; ++i) {
    for (j = 0; j < A_cols; ++j) {
      printf("%f, ", A[i * A_cols + j]);
    }
  }
  printf("]\n");
}

int main(int argc, char *argv[]) {

  // Assign seed from current time integer
  time_t t;
  srand((unsigned int)time(&t));

  int64_t A_rows, A_cols, B_rows, B_cols;
  int32_t steps = 1;

  if (argc == 5 || argc == 4) {
    A_rows = atoll(argv[1]);
    A_cols = atoll(argv[2]);
    B_rows = atoll(argv[2]);
    B_cols = atoll(argv[3]);
    if (argc == 5) {
      steps = atoll(argv[4]);
    }
  } else {
    printf("Usage: \n"
           "- 3 arguments: matrix A rows, matrix A cols and matrix B cols\n"
           "- 4 arguments: matrix A rows, matrix A cols and matrix B cols, "
           "steps\n");
    return 1;
  }

  printf("[ %ld %ld %ld ]\n", A_rows, A_cols, B_cols);

  struct timespec start, tmp;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  double *A =
      (double *)malloc((size_t)A_rows * (size_t)A_cols * sizeof(double));
  tmp = print_dtime(start, "allocate A");

  double *B =
      (double *)malloc((size_t)B_rows * (size_t)B_cols * sizeof(double));
  tmp = print_dtime(tmp, "allocate B");

  // value-init to zero
  double *C = (double *)calloc((size_t)A_rows * (size_t)B_cols, sizeof(double));
  tmp = print_dtime(tmp, "initialize C");

  fill_random(A, A_rows, A_cols);
  tmp = print_dtime(tmp, "fill A");

  fill_random(B, B_rows, B_cols);
  tmp = print_dtime(tmp, "fill B");

  int32_t i;
  gemm(A, B, C, A_rows, A_cols, B_cols);
  tmp = print_dtime(tmp, "blas gemm");

  if (steps > 1) {

    double average_time = 0;
    struct timespec start_i, end_i;

    for (i = 1; i < steps; ++i) {
      clock_gettime(CLOCK_MONOTONIC_RAW, &start_i);
      gemm(A, B, C, A_rows, A_cols, B_cols);
      end_i = print_dtime(start_i, "blas gemm");
      average_time += dtime(start_i, end_i);
    }
    average_time /= (steps - 1);
    const double gflops =
        (double)((2 * A_rows * A_cols * B_cols * 1E-9) / average_time);
    printf("GFLOPS: %lf  steps: %d average_time: %f\n", gflops, steps,
           average_time);
  }

  print_dtime(start, "total time");

  free(A);
  free(B);
  free(C);

  return 0;
}
