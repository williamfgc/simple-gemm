#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int BLOCK_SIZE = 16;

static void fill_random(float *A, const int64_t n, const int64_t m) {

  int64_t i, j;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < m; ++j) {
      A[i * m + j] = (float)rand() / (float)RAND_MAX;
    }
  }
}

__global__ 
void gemm(float *A, float *B, float *C, const int64_t A_rows,
                     const int64_t A_cols, const int64_t B_cols) {

    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
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

int main(int argc, char *argv[]) {

  // Assign seed from current time integer
  time_t t;
  srand((unsigned int)time(&t));

  int64_t A_rows, A_cols, B_rows, B_cols;

  if (argc != 4) {
    printf(
        "Usage: 3 arguments: matrix A rows, matrix A cols and matrix B cols\n");
    return 1;
  } else {
    A_rows = atoll(argv[1]);
    A_cols = atoll(argv[2]);
    B_rows = atoll(argv[2]);
    B_cols = atoll(argv[3]);
  }

  printf("[ %ld %ld %ld ]\n", A_rows, A_cols, B_cols);

  struct timespec start, tmp;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  float* A_h, B_h, C_h;
  
  cudaMallocHost( ( void**) &A_h, sizeof(float) * A_rows * A_cols);
  tmp = print_dtime(start, "allocate A");

  cudaMallocHost( ( void**) &B_h, sizeof(float) * B_rows * B_cols);
  tmp = print_dtime(tmp, "allocate B");

  cudaMallocHost( ( void**) &C_h, sizeof(float) * A_rows * B_cols);
  tmp = print_dtime(tmp, "allocate C"); 
  
  // value-init to zero
  fill_random(A_h, A_rows, A_cols);
  tmp = print_dtime(tmp, "fill A");

  fill_random(B_h, B_rows, B_cols);
  tmp = print_dtime(tmp, "fill B");

  
  // Allocate memory space on the device
  double *A_d, *B_d, *C_d;
  cudaMalloc((void **) &A_d, sizeof(float) * A_rows * A_cols );
  tmp = print_dtime(tmp, "allocate A_d");
  cudaMalloc((void **) &B_d, sizeof(float) * B_rows * B_cols );
  tmp = print_dtime(tmp, "allocate B_d");
  cudaMalloc((void **) &C_d, sizeof(float) * A_rows * B_cols );
  tmp = print_dtime(tmp, "allocate C_d");

  cudaMemcpy(A_d, A_h, sizeof(float)*A_rows*A_cols, cudaMemcpuHostToDevice);
  tmp = print_dtime(tmp, "copy A");

  cudaMemcpy(B_d, B_h, sizeof(float)*B_rows*B_cols, cudaMemcpuHostToDevice);
  tmp = print_dtime(tmp, "copy B");
 
  unsigned int grid_rows = (A_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_cols = (B_cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  gemm <<<dimGrid, dimBlock>>>( A_d, B_d, C_d, A_rows, A_cols, B_cols);
  tmp = print_dtime(tmp, "simple gemm");

  cudaMemcpy(C_h, C_d, sizeof(float)*A_rows*B_cols, cudaMemcpuDeviceToHost);
  tmp = print_dtime(tmp, "copy C");

  print_dtime(start, "total time");

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  free(A_h);
  free(B_h);
  free(C_h);

  return 0;
}
