

Julia version of simple gemm

Requirements: recent julia [v1.7.2](https://julialang.org/downloads/#current_stable_release)

Steps:

1. Instatiate dependencies from `simple-gemm.jl` directory:

```
    $ julia --project=. 
    julia> ]
    (simple-gemm) pkg> instantiate
```

2. Run example from terminal:

```
    $ julia src/gemm-dense.jl 10 10 10
```

3. Modify `src/gemm-dense.jl` to generate llvm ir, native or time instrumentations
