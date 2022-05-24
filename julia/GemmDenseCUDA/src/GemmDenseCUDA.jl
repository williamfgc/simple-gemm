module GemmDenseCUDA

import CUDA

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
    A = CUDA.CuArray{Float32,2}(undef, A_rows, A_cols)
    B = CUDA.CuArray{Float32,2}(undef, B_rows, B_cols)
    C = CUDA.zeros(Float32, A_rows, B_cols)

    CUDA.rand!(A)
    CUDA.rand!(B)

    # uses cuBLAS 
    C = A * B
    CUDA.synchronize()

    return 0

end


end
