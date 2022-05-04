module GemmDenseThreads

import Random

@doc """
Simplified gemm: C = alpha A x B + C where alpha = 1 , C = zeros, 
so:
C = A x B 
"""
function gemm!(A::Array{Float32,2}, B::Array{Float32,2}, C::Array{Float32,2})

    A_rows = size(A)[1]
    A_cols = size(A)[2]
    B_cols = size(B)[2]


    Threads.@threads for j = 1:B_cols
        for l = 1:A_cols
            @inbounds temp::Float32 = B[l, j]::Float32
            for i = 1:A_rows
                @inbounds C[i, j] += temp * A[i, l]
            end
        end
    end



end

function main(args::Array{String,1})::Int32

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
    print("Time to allocate A ")
    @time A = Array{Float32,2}(undef, A_rows, A_cols)
    print("Time to allocate B ")
    @time B = Array{Float32,2}(undef, B_rows, B_cols)
    print("Time to initialize C ")
    @time C = zeros(Float32, A_rows, B_cols)

    print("Time to fill A ")
    @time Random.rand!(A)
    print("Time to fill B ")
    @time Random.rand!(B)

    print("Time to simple gemm ")
    @time gemm!(A, B, C)

    # println(C)

    return 0

end

end # module
