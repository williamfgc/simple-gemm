# simple-gemm
Collection of simple General Matrix Multiplication - GEMM implementations

```
C = a . A x B + C
if a = 1 and C = zeros
C = A x B
```

A and B are initialized with random numbers
C is initialized with zeros

Arguments are always 3 matrix dimensions: `args = [A_rows, A_cols (= B_rows), B_cols]` 

*e.g.* 5 5 5 or 10 10 10

CPU multithreading:

- `GemmDenseThreads`: native Julia Threads implementation

    ```
    $ cd GemmDenseThreads
    $ julia -t 4 gemm-dense-threads.jl 5 5 5    
    ```

- `GemmDenseBlas`: uses `LinearAlgebra.jl` (super-fast), if compiled with `OpenBLAS` set `OPENBLAS_NUM_THREADS` 

    ```
    $ cd GemmDenseThreads
    $ OPENBLAS_NUM_THREADS=4 julia gemm-dense-blas.jl 5 5 5    
    ```

GPU :

- `GemmDenseCUDA` : uses `CUDA.jl` which uses the optimized `cuBLAS` (very fast) on NVIDIA GPUs

    ```
    $ cd GemmDenseCUDA
    $ julia gemm-dense-cuda.jl 5 5 5
    ```
