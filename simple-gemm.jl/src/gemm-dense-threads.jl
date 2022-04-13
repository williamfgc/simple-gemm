
import Random
import Dates
import BenchmarkTools
import InteractiveUtils

@doc """
Generate random values from global random number generator
"""
function fill_random(A::Array{Float32,2})
    return Random.rand!(A)
end

@doc """
Simplified gemm: C = alpha A x B + C where alpha = 1 , C = zeros, 
so:
C = A x B 
"""
function gemm!(A::Array{Float32,2}, B::Array{Float32,2}, C::Array{Float32,2})
    A_rows = size(A)[2]
    A_cols = size(A)[1]
    B_rows = size(B)[2]


    Base.Threads.@threads for i = 1:A_rows
        for k = 1:A_cols
            for j = 1:B_rows
                C[j, i] += A[k, i] * B[j, k]
            end
        end
    end

    return
end

function main(args::Array{String,1})

    # must initialize scalars
    A_rows::Int32 = -1
    A_cols::Int32 = -1
    B_rows::Int32 = -1
    B_cols::Int32 = -1

    @show args

    # args don't include Julia executable and program
    if size(args)[1] != 3
        throw(
            ArgumentError(
                "Usage: 3 arguments: matrix A rows, matrix A cols and matrix B cols",
            ),
        )
    else
        A_rows = parse(Int32, args[1])
        A_cols = parse(Int32, args[2])
        B_rows = parse(Int32, args[2])
        B_cols = parse(Int32, args[3])
    end

    # Julia is column-based (like Fortran)
    A = Array{Float32,2}(undef, A_cols, A_rows)
    B = Array{Float32,2}(undef, B_cols, B_rows)
    C = zeros(Float32, B_cols, A_rows)

    fill_random(A)
    fill_random(B)

    BenchmarkTools.@time gemm!(A, B, C)
    #InteractiveUtils.@code_llvm gemm!(A, B, C)
    #InteractiveUtils.@code_native gemm!(A, B, C)


    # println(A)
    # println(B)
    # println(C)

end

main(ARGS)
