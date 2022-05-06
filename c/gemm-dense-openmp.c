#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static void fill_random(float *A, const int n, const int m) {

  int i, j;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < m; ++j) {
      A[i * m + j] = (float)rand() / (float)RAND_MAX;
    }
  }
}

static void gemm(float *A, float *B, float *C, const int A_rows,
                 const int A_cols, const int B_rows) {
  int i, k, j;
  float temp;
#ifdef _OPENMP
#pragma omp parallel for shared(A, B, C, A_rows, A_cols,                       \
                                B_rows) private(i, k, j, temp)
#endif
  for (i = 0; i < A_rows; i++) {
    for (k = 0; k < A_cols; k++) {
      temp = A[i * A_cols + k];
      for (j = 0; j < B_rows; j++) {
        C[i * B_rows + j] += temp * B[k * B_rows + j];
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

static void print_matrix(float *A, const int A_rows, const int A_cols) {

  int i, j;
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

  int A_rows, A_cols, B_rows, B_cols;

  if (argc != 4) {
    printf(
        "Usage: 3 arguments: matrix A rows, matrix A cols and matrix B cols\n");
    return 1;
  } else {
    A_rows = atoi(argv[1]);
    A_cols = atoi(argv[2]);
    B_rows = atoi(argv[2]);
    B_cols = atoi(argv[3]);
  }

  struct timespec start, tmp;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  float *A = (float *)malloc(A_rows * A_cols * sizeof(float));
  tmp = print_dtime(start, "allocate A");

  float *B = (float *)malloc(B_rows * B_cols * sizeof(float));
  tmp = print_dtime(tmp, "allocate B");

  // value-init to zero
  float *C = (float *)calloc(A_rows * B_cols, sizeof(float));
  tmp = print_dtime(tmp, "initialize C");

  fill_random(A, A_rows, A_cols);
  tmp = print_dtime(tmp, "fill A");

  fill_random(B, B_rows, B_cols);
  tmp = print_dtime(tmp, "fill B");

  gemm(A, B, C, A_rows, A_cols, B_cols);
  tmp = print_dtime(tmp, "simple gemm");

  print_dtime(start, "total time");

  free(A);
  free(B);
  free(C);

  return 0;
}
