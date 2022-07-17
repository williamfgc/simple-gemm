#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cublas_v2.h>

static int BLOCK_SIZE = 32;

static void fill_random(double *A, const int64_t n, const int64_t m) {

  int64_t i, j;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < m; ++j) {
      A[i * m + j] = (double)rand() / (double)RAND_MAX;
    }
  }
}

__global__ void gemm(double *A, double *B, double *C, int64_t A_rows,
                     int64_t A_cols, int64_t B_cols) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  double sum = 0.0f;
  int i;

  if (col < B_cols && row < A_rows) {
    for (i = 0; i < A_cols; ++i) {
      sum += A[row * A_cols + i] * B[i * B_cols + col];
    }
    C[row * B_cols + col] = sum;
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
  if (cudaMalloc((void **)&A_d, sizeof(double) * A_rows * A_cols)) {
    printf("A_d allocation failure\n");
    exit(1); // leaky exit
  }
  tmp = print_dtime(tmp, "allocate A_d");

  if (cudaMalloc((void **)&B_d, sizeof(double) * B_rows * B_cols)) {
    printf("B_d allocation failure\n");
    exit(1); // leaky exit
  }
  tmp = print_dtime(tmp, "allocate B_d");

  if (cudaMalloc((void **)&C_d, sizeof(double) * A_rows * B_cols)) {
    printf("C_d allocation failure\n");
    exit(1); // leaky exit
  }
  tmp = print_dtime(tmp, "allocate C_d");

  cudaMemcpy(A_d, A_h, sizeof(double) * A_rows * A_cols,
             cudaMemcpyHostToDevice);
  tmp = print_dtime(tmp, "copy A");

  cudaMemcpy(B_d, B_h, sizeof(double) * B_rows * B_cols,
             cudaMemcpyHostToDevice);
  tmp = print_dtime(tmp, "copy B");

  unsigned int grid_rows = (A_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_cols = (B_cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  int32_t i;
  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A_rows, A_cols, B_cols, 
     1.0, A_d, A_rows, B_d, B_cols, 0., C_d, A_rows);
  cudaDeviceSynchronize();
  tmp = print_dtime(tmp, "cublas gemm");

  if (steps > 1) {

    double average_time = 0;
    struct timespec start_i, end_i;

    for (i = 1; i < steps; ++i) {
      clock_gettime(CLOCK_MONOTONIC_RAW, &start_i);
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A_rows, A_cols, B_cols, 
                  1.0, A_d, A_rows, B_d, B_cols, 0., C_d, A_rows);
      cudaDeviceSynchronize();
      end_i = print_dtime(start_i, "cublas gemm");
      average_time += dtime(start_i, end_i);
    }
    average_time /= (steps - 1);
    const double gflops =
        (double)((2 * A_rows * A_cols * B_cols * 1E-9) / average_time);
    printf("GFLOPS: %lf  steps: %d average_time: %f\n", gflops, steps,
           average_time);
  }

  cudaMemcpy(C_h, C_d, sizeof(double) * A_rows * B_cols,
             cudaMemcpyDeviceToHost);
  tmp = print_dtime(tmp, "copy C");

  // print_matrix(A_h, A_rows, A_cols);
  // print_matrix(B_h, B_rows, B_cols);
  // print_matrix(C_h, A_rows, B_cols);

  print_dtime(start, "total time");

  cublasDestroy(handle);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  free(A_h);
  free(B_h);
  free(C_h);

  return 0;
}
