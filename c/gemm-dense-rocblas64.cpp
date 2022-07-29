#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "hip/hip_runtime.h"
#include "rocblas/rocblas.h"

static void fill_random(double *A, const int64_t n, const int64_t m) {

  int64_t i, j;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < m; ++j) {
      A[i * m + j] = (double)rand() / (double)RAND_MAX;
    }
  }
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

static double dtime(struct timespec start, struct timespec end) {
  return ((double)((end.tv_sec - start.tv_sec) * 1000000 +
                   (end.tv_nsec - start.tv_nsec) / 1000)) /
         1E6;
}

int main(int argc, char *argv[]) {

  // Assign seed from hiprrent time integer
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

  double *A_h =
      (double *)malloc((size_t)A_rows * (size_t)A_cols * sizeof(double));
  tmp = print_dtime(start, "allocate A");

  double *B_h =
      (double *)malloc((size_t)B_rows * (size_t)B_cols * sizeof(double));
  tmp = print_dtime(tmp, "allocate B");

  double *C_h =
      (double *)malloc((size_t)A_rows * (size_t)B_cols * sizeof(double));
  tmp = print_dtime(tmp, "allocate C");

  // value-init to zero
  fill_random(A_h, A_rows, A_cols);
  tmp = print_dtime(tmp, "fill A");

  fill_random(B_h, B_rows, B_cols);
  tmp = print_dtime(tmp, "fill B");

  // Allocate memory space on the device
  double *A_d, *B_d, *C_d;
  if (hipMalloc((void **)&A_d, sizeof(double) * A_rows * A_cols)) {
    printf("A_d allocation failure\n");
    exit(1); // leaky exit
  }
  tmp = print_dtime(tmp, "allocate A_d");

  if (hipMalloc((void **)&B_d, sizeof(double) * B_rows * B_cols)) {
    printf("B_d allocation failure\n");
    exit(1); // leaky exit
  }
  tmp = print_dtime(tmp, "allocate B_d");

  if (hipMalloc((void **)&C_d, sizeof(double) * A_rows * B_cols)) {
    printf("C_d allocation failure\n");
    exit(1); // leaky exit
  }
  tmp = print_dtime(tmp, "allocate C_d");

  hipMemcpy(A_d, A_h, sizeof(double) * A_rows * A_cols, hipMemcpyHostToDevice);
  tmp = print_dtime(tmp, "copy A");

  hipMemcpy(B_d, B_h, sizeof(double) * B_rows * B_cols, hipMemcpyHostToDevice);
  tmp = print_dtime(tmp, "copy B");

  int32_t i;
  // Create a handle for rocBLAS
  rocblas_handle handle;
  rocblas_create_handle(&handle);

  const double alpha = 1.;
  const double beta = 0.;

  rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none, A_rows,
                A_cols, B_cols, &alpha, A_d, A_rows, B_d, B_cols, &beta, C_d,
                A_rows);
  hipDeviceSynchronize();
  tmp = print_dtime(tmp, "hipblas gemm");

  if (steps > 1) {

    double average_time = 0;
    struct timespec start_i, end_i;

    for (i = 1; i < steps; ++i) {
      clock_gettime(CLOCK_MONOTONIC_RAW, &start_i);
      rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                    A_rows, A_cols, B_cols, &alpha, A_d, A_rows, B_d, B_cols,
                    &beta, C_d, A_rows);
      hipDeviceSynchronize();
      end_i = print_dtime(start_i, "hipblas gemm");
      average_time += dtime(start_i, end_i);
    }
    average_time /= (steps - 1);
    const double gflops =
        (double)((2 * A_rows * A_cols * B_cols * 1E-9) / average_time);
    printf("GFLOPS: %lf  steps: %d average_time: %f\n", gflops, steps,
           average_time);
  }

  hipMemcpy(C_h, C_d, sizeof(double) * A_rows * B_cols, hipMemcpyDeviceToHost);
  tmp = print_dtime(tmp, "copy C");

  // print_matrix(A_h, A_rows, A_cols);
  // print_matrix(B_h, B_rows, B_cols);
  // print_matrix(C_h, A_rows, B_cols);

  print_dtime(start, "total time");

  rocblas_destroy_handle(handle);
  hipFree(A_d);
  hipFree(B_d);
  hipFree(C_d);

  free(A_h);
  free(B_h);
  free(C_h);

  return 0;
}
