#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int BLOCK_SIZE = 32;

static void fill_random(float *A, const int64_t n, const int64_t m) {

  int64_t i, j;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < m; ++j) {
      A[i * m + j] = (float)rand() / (float)RAND_MAX;
    }
  }
}

__global__ void gemm(float *A, float *B, float *C, int64_t A_rows,
                     int64_t A_cols, int64_t B_cols) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
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
  const float dtime = ((float)((end.tv_sec - start.tv_sec) * 1000000 +
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

static float dtime(struct timespec start, struct timespec end) {
  return ((float)((end.tv_sec - start.tv_sec) * 1000000 +
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
  if (cudaMalloc((void **)&A_d, sizeof(float) * A_rows * A_cols)) {
    printf("A_d allocation failure\n");
    exit(1); // leaky exit
  }
  tmp = print_dtime(tmp, "allocate A_d");

  if (cudaMalloc((void **)&B_d, sizeof(float) * B_rows * B_cols)) {
    printf("B_d allocation failure\n");
    exit(1); // leaky exit
  }
  tmp = print_dtime(tmp, "allocate B_d");

  if (cudaMalloc((void **)&C_d, sizeof(float) * A_rows * B_cols)) {
    printf("C_d allocation failure\n");
    exit(1); // leaky exit
  }
  tmp = print_dtime(tmp, "allocate C_d");

  cudaMemcpy(A_d, A_h, sizeof(float) * A_rows * A_cols, cudaMemcpyHostToDevice);
  tmp = print_dtime(tmp, "copy A");

  cudaMemcpy(B_d, B_h, sizeof(float) * B_rows * B_cols, cudaMemcpyHostToDevice);
  tmp = print_dtime(tmp, "copy B");

  unsigned int grid_rows = (A_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_cols = (B_cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  int32_t i;
  gemm<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, A_rows, A_cols, B_cols);
  cudaDeviceSynchronize();
  tmp = print_dtime(tmp, "simple gemm");

  if (steps > 1) {

    float average_time = 0;
    struct timespec start_i, end_i;

    for (i = 1; i < steps; ++i) {
      clock_gettime(CLOCK_MONOTONIC_RAW, &start_i);
      gemm<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, A_rows, A_cols, B_cols);
      cudaDeviceSynchronize();
      end_i = print_dtime(start_i, "simple gemm");
      average_time += dtime(start_i, end_i);
    }
    average_time /= (steps - 1);
    const double gflops =
        (double)((2 * A_rows * A_cols * B_cols * 1E-9) / average_time);
    printf("GFLOPS: %lf  steps: %d ", gflops, steps);
  }

  cudaMemcpy(C_h, C_d, sizeof(float) * A_rows * B_cols, cudaMemcpyDeviceToHost);
  tmp = print_dtime(tmp, "copy C");

  // print_matrix(A_h, A_rows, A_cols);
  // print_matrix(B_h, B_rows, B_cols);
  // print_matrix(C_h, A_rows, B_cols);

  print_dtime(start, "total time");

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  free(A_h);
  free(B_h);
  free(C_h);

  return 0;
}
