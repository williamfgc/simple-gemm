# GemmDenseCUDA

Hand-rolled Julia implementation of a simple gemm C = A x B matrix multiplication on GPU using [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)

To run from this directory:

- Install required packages:

```
$ julia --project
julia> ]
GemmDenseCUDA> instantiate
...
GemmDenseCUDA> <- (backspace)
julia> exit()
```

- Run gemm-dense-cuda.jl for a M = N = 10,000 system 

```
$ julia --project gemm-dense-cuda.jl 10000 10000 10000
```

- Run gemm-dense-cuda.jl for a M = N = 10,000 system with 5 repetitions

```
$ julia --project gemm-dense-cuda.jl 10000 10000 10000 5
```

Optionally if start-up times are an issue, we can precompile the dependencies into a shared library `GemmDenseCUDA_jll.so` using the PackageCompiler.jl script:

```
$ julia --project script/precompile.jl
```

and rerun cases:

```
$ julia --project -JGemmDenseCUDA_jll.so gemm-dense-cuda.jl 10000 10000 10000

$ julia --project -JGemmDenseCUDA_jll.so gemm-dense-cuda.jl 10000 10000 10000 5
```




