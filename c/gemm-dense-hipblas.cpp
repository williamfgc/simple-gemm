#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "hip/hip_runtime.h"
#include "hipblas.h"

static int BLOCK_SIZE = 32;

static void fill_random(float *A, const int64_t n, const int64_t m) {

  int64_t i, j;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < m; ++j) {
      A[i * m + j] = (float)rand() / (float)RAND_MAX;
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

  float *A_h = (float *)malloc((size_t)A_rows * (size_t)A_cols * sizeof(float));
  tmp = print_dtime(start, "allocate A");

  float *B_h = (float *)malloc((size_t)B_rows * (size_t)B_cols * sizeof(float));
  tmp = print_dtime(tmp, "allocate B");

  float *C_h = (float *)malloc((size_t)A_rows * (size_t)B_cols * sizeof(float));
  tmp = print_dtime(tmp, "allocate C");

  // value-init to zero
  fill_random(A_h, A_rows, A_cols);
  tmp = print_dtime(tmp, "fill A");

  fill_random(B_h, B_rows, B_cols);
  tmp = print_dtime(tmp, "fill B");

  // Allocate memory space on the device
  float *A_d, *B_d, *C_d;
  if (hipdaMalloc((void **)&A_d, sizeof(float) * A_rows * A_cols)) {
    printf("A_d allocation failure\n");
    exit(1); // leaky exit
  }
  tmp = print_dtime(tmp, "allocate A_d");

  if (hipdaMalloc((void **)&B_d, sizeof(float) * B_rows * B_cols)) {
    printf("B_d allocation failure\n");
    exit(1); // leaky exit
  }
  tmp = print_dtime(tmp, "allocate B_d");

  if (hipdaMalloc((void **)&C_d, sizeof(float) * A_rows * B_cols)) {
    printf("C_d allocation failure\n");
    exit(1); // leaky exit
  }
  tmp = print_dtime(tmp, "allocate C_d");

  hipdaMemcpy(A_d, A_h, sizeof(float) * A_rows * A_cols,
              hipdaMemcpyHostToDevice);
  tmp = print_dtime(tmp, "copy A");

  hipdaMemcpy(B_d, B_h, sizeof(float) * B_rows * B_cols,
              hipdaMemcpyHostToDevice);
  tmp = print_dtime(tmp, "copy B");

  unsigned int grid_rows = (A_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_cols = (B_cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  int32_t i;
  // Create a handle for hipBLAS
  hipblasHandle_t handle;
  hipblasCreate(&handle);
  const float alpha = 1.;
  const float beta = 0.;

  hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, A_rows, A_cols, B_cols,
               &alpha, A_d, A_rows, B_d, B_cols, &beta, C_d, A_rows);
  hipDeviceSynchronize();
  tmp = print_dtime(tmp, "hipblas gemm");

  if (steps > 1) {

    double average_time = 0;
    struct timespec start_i, end_i;

    for (i = 1; i < steps; ++i) {
      clock_gettime(CLOCK_MONOTONIC_RAW, &start_i);
      hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, A_rows, A_cols, B_cols,
                   &alpha, A_d, A_rows, B_d, B_cols, &beta, C_d, A_rows);
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

  hipMemcpy(C_h, C_d, sizeof(float) * A_rows * B_cols, hipMemcpyDeviceToHost);
  tmp = print_dtime(tmp, "copy C");

  // print_matrix(A_h, A_rows, A_cols);
  // print_matrix(B_h, B_rows, B_cols);
  // print_matrix(C_h, A_rows, B_cols);

  print_dtime(start, "total time");

  hipblasDestroy(handle);
  hipFree(A_d);
  hipFree(B_d);
  hipFree(C_d);

  free(A_h);
  free(B_h);
  free(C_h);

  return 0;
}
