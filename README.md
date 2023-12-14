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

- `GemmDenseThreads.py`: native Python Numba Threads implementation

    ```
    $ cd python/GemmDenseThreads
    $ NUMBA_NUM_THREADS=4 python3 GemmDenseThreads.py 5 5 5    
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
    
# Citation
If you find the repository useful, please cite the reference [2023 IPDPSW paper](https://doi.org/10.1109/IPDPSW59300.2023.00068):

```
@INPROCEEDINGS{10196600,
  author={Godoy, William F. and Valero-Lara, Pedro and Dettling, T. Elise and Trefftz, Christian and Jorquera, Ian and Sheehy, Thomas and Miller, Ross G. and Gonzalez-Tallada, Marc and Vetter, Jeffrey S. and Churavy, Valentin},
  booktitle={2023 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)}, 
  title={Evaluating performance and portability of high-level programming models: Julia, Python/Numba, and Kokkos on exascale nodes}, 
  year={2023},
  volume={},
  number={},
  pages={373-382},
  doi={10.1109/IPDPSW59300.2023.00068}}
```
