#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static void fill_random(_Float16 *A, const int64_t n, const int64_t m) {

  int64_t i, j;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < m; ++j) {
      A[i * m + j] = (_Float16)((float)rand() / (float)RAND_MAX);
    }
  }
}

static void gemm(_Float16 *A, _Float16 *B, float *C, const int64_t A_rows,
                 const int64_t A_cols, const int64_t B_cols) {
  int64_t i, k, j;
  _Float16 temp;
#ifdef _OPENMP
#pragma omp parallel for default(shared) private(i, k, j, temp)
#endif
  for (i = 0; i < A_rows; i++) {
    for (k = 0; k < A_cols; k++) {
      temp = A[i * A_cols + k];
      for (j = 0; j < B_cols; j++) {
        C[i * B_cols + j] += temp * B[k * B_cols + j];
      }
    }
  }
}

static struct timespec print_dtime(struct timespec start, const char *process) {
  struct timespec end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  const float dtime = ((float)((end.tv_sec - start.tv_sec) * 1000000 +
                               (end.tv_nsec - start.tv_nsec) / 1000)) /
                      1E6;

  printf("Time to %s = %f s\n", process, dtime);
  return end;
}

static float dtime(struct timespec start, struct timespec end) {
  return ((float)((end.tv_sec - start.tv_sec) * 1000000 +
                  (end.tv_nsec - start.tv_nsec) / 1000)) /
         1E6;
}

static void print_matrix(float *A, const int64_t A_rows, const int64_t A_cols) {

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

  _Float16 *A =
      (_Float16 *)malloc((size_t)A_rows * (size_t)A_cols * sizeof(_Float16));
  tmp = print_dtime(start, "allocate A");

  _Float16 *B =
      (_Float16 *)malloc((size_t)B_rows * (size_t)B_cols * sizeof(_Float16));
  tmp = print_dtime(tmp, "allocate B");

  // value-init to zero
  float *C = (float *)calloc((size_t)A_rows * (size_t)B_cols, sizeof(float));
  tmp = print_dtime(tmp, "initialize C");

  fill_random(A, A_rows, A_cols);
  tmp = print_dtime(tmp, "fill A");

  fill_random(B, B_rows, B_cols);
  tmp = print_dtime(tmp, "fill B");

  int32_t i;
  gemm(A, B, C, A_rows, A_cols, B_cols);
  tmp = print_dtime(tmp, "simple gemm");

  if (steps > 1) {

    float average_time = 0;
    struct timespec start_i, end_i;

    for (i = 1; i < steps; ++i) {
      clock_gettime(CLOCK_MONOTONIC_RAW, &start_i);
      gemm(A, B, C, A_rows, A_cols, B_cols);
      end_i = print_dtime(start_i, "simple gemm");
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
