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

function gemm64!(A::Array{Float64,2}, B::Array{Float64,2}, C::Array{Float64,2})

    A_rows = size(A)[1]
    A_cols = size(A)[2]
    B_cols = size(B)[2]

    Threads.@threads for j = 1:B_cols
        for l = 1:A_cols
            @inbounds temp::Float64 = B[l, j]::Float64
            for i = 1:A_rows
                @inbounds C[i, j] += temp * A[i, l]
            end
        end
    end
end

function gemm16!(A::Array{Float16,2}, B::Array{Float16,2}, C::Array{Float32,2})

    A_rows = size(A)[1]
    A_cols = size(A)[2]
    B_cols = size(B)[2]

    Threads.@threads for j = 1:B_cols
        for l = 1:A_cols
            @inbounds temp::Float16 = B[l, j]::Float16
            for i = 1:A_rows
                @inbounds C[i, j] += temp * A[i, l]
            end
        end
    end
end

function main(args::Array{String,1})::Int32

    # must initialize scalars
    A_rows::Int64 = -1
    A_cols::Int64 = -1
    B_rows::Int64 = -1
    B_cols::Int64 = -1

    steps::Int32 = 1

    @show args

    # args don't include Julia executable and program
    nargs = size(args)[1]

    if nargs == 4 || nargs == 3
        A_rows = parse(Int64, args[1])
        A_cols = parse(Int64, args[2])
        B_rows = parse(Int64, args[2])
        B_cols = parse(Int64, args[3])

        if nargs == 4
            steps = parse(Int32, args[4])
        end
    else
        throw(
            ArgumentError(
                "Usage: \n
                  - 3 arguments: matrix A rows, matrix A cols and matrix B cols\n
                  - 4 arguments: matrix A rows, matrix A cols and matrix B cols, steps",
            ),
        )
    end

    # Julia is column-based (like Fortran)
    @time begin

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

        if steps > 1
            timings = zeros(steps)
            for i = 2:steps
                print("Time to simple gemm ")
                timings[i] = @elapsed gemm!(A, B, C)
                println(timings[i])
            end

            average_time = sum(timings) / (steps - 1)
            gflops = (2 * A_rows * A_cols * B_cols) * 1E-9 / average_time
            println("GFLOPS: ", gflops, " steps: ", steps)
        end

        print("Time to total time ")
    end
    return 0

end

function main64(args::Array{String,1})::Int32

    # must initialize scalars
    A_rows::Int64 = -1
    A_cols::Int64 = -1
    B_rows::Int64 = -1
    B_cols::Int64 = -1

    steps::Int32 = 1

    @show args

    # args don't include Julia executable and program
    nargs = size(args)[1]

    if nargs == 4 || nargs == 3
        A_rows = parse(Int64, args[1])
        A_cols = parse(Int64, args[2])
        B_rows = parse(Int64, args[2])
        B_cols = parse(Int64, args[3])

        if nargs == 4
            steps = parse(Int32, args[4])
        end
    else
        throw(
            ArgumentError(
                "Usage: \n
                  - 3 arguments: matrix A rows, matrix A cols and matrix B cols\n
                  - 4 arguments: matrix A rows, matrix A cols and matrix B cols, steps",
            ),
        )
    end

    # Julia is column-based (like Fortran)
    @time begin

        print("Time to allocate A ")
        @time A = Array{Float64,2}(undef, A_rows, A_cols)
        print("Time to allocate B ")
        @time B = Array{Float64,2}(undef, B_rows, B_cols)
        print("Time to initialize C ")
        @time C = zeros(Float64, A_rows, B_cols)

        print("Time to fill A ")
        @time Random.rand!(A)
        print("Time to fill B ")
        @time Random.rand!(B)

        print("Time to simple gemm ")
        @time gemm64!(A, B, C)

        if steps > 1
            timings = zeros(steps)
            for i = 2:steps
                print("Time to simple gemm ")
                timings[i] = @elapsed gemm64!(A, B, C)
                println(timings[i])
            end

            average_time = sum(timings) / (steps - 1)
            gflops = (2 * A_rows * A_cols * B_cols) * 1E-9 / average_time
            println("GFLOPS: ", gflops, " steps: ", steps)
        end

        print("Time to total time ")
    end
    return 0

end

function main16(args::Array{String,1})::Int32

    # must initialize scalars
    A_rows::Int64 = -1
    A_cols::Int64 = -1
    B_rows::Int64 = -1
    B_cols::Int64 = -1

    steps::Int32 = 1

    @show args

    # args don't include Julia executable and program
    nargs = size(args)[1]

    if nargs == 4 || nargs == 3
        A_rows = parse(Int64, args[1])
        A_cols = parse(Int64, args[2])
        B_rows = parse(Int64, args[2])
        B_cols = parse(Int64, args[3])

        if nargs == 4
            steps = parse(Int32, args[4])
        end
    else
        throw(
            ArgumentError(
                "Usage: \n
                  - 3 arguments: matrix A rows, matrix A cols and matrix B cols\n
                  - 4 arguments: matrix A rows, matrix A cols and matrix B cols, steps",
            ),
        )
    end

    # Julia is column-based (like Fortran)
    @time begin

        print("Time to allocate A ")
        @time A = Array{Float16,2}(undef, A_rows, A_cols)
        print("Time to allocate B ")
        @time B = Array{Float16,2}(undef, B_rows, B_cols)
        print("Time to initialize C ")
        @time C = zeros(Float32, A_rows, B_cols)

        print("Time to fill A ")
        @time Random.rand!(A)
        print("Time to fill B ")
        @time Random.rand!(B)

        print("Time to simple gemm ")
        @time gemm16!(A, B, C)

        if steps > 1
            timings = zeros(steps)
            for i = 2:steps
                print("Time to simple gemm ")
                timings[i] = @elapsed gemm16!(A, B, C)
                println(timings[i])
            end

            average_time = sum(timings) / (steps - 1)
            gflops = (2 * A_rows * A_cols * B_cols) * 1E-9 / average_time
            println("GFLOPS: ", gflops, " steps: ", steps)
        end

        print("Time to total time ")
    end
    return 0

end

end # module
