module GemmDenseCUBLAS

import CUDA

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

    @time begin

        # Julia is column-based (like Fortran)
        print("Time to allocate A")
        @time A = CUDA.CuArray{Float32,2}(undef, A_rows, A_cols)

        print("Time to allocate B")
        @time B = CUDA.CuArray{Float32,2}(undef, B_rows, B_cols)

        print("Time to initialize C")
        @time C = CUDA.zeros(Float32, A_rows, B_cols)

        print("Time to fill A")
        @time CUDA.rand!(A)
        print("Time to fill B")
        @time CUDA.rand!(B)

        print("Time to simple gemm ")
        @time begin
            C = A * B
            CUDA.synchronize()
        end

        if steps > 1
            timings = zeros(steps)
            for i = 2:steps
                print("Time to simple gemm ")
                timings[i] = @elapsed begin
                    C = A * B
                    CUDA.synchronize()
                end

                println(timings[i])
            end

            average_time = sum(timings) / (steps - 1)
            gflops = (2 * A_rows * A_cols * B_cols) * 1E-9 / average_time
            println("GFLOPS: ", gflops, " steps: ", steps)
        end

        print("Time to total time ")
    end

    return 0
end #main

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

    @time begin

        # Julia is column-based (like Fortran)
        print("Time to allocate A")
        @time A = CUDA.CuArray{Float64,2}(undef, A_rows, A_cols)

        print("Time to allocate B")
        @time B = CUDA.CuArray{Float64,2}(undef, B_rows, B_cols)

        print("Time to initialize C")
        @time C = CUDA.zeros(Float64, A_rows, B_cols)

        print("Time to fill A")
        @time CUDA.rand!(A)
        print("Time to fill B")
        @time CUDA.rand!(B)

        # println("Threads: ", threads)
        # println("Blocks: ", blocks)

        print("Time to simple gemm ")
        @time begin
            C = A * B
            CUDA.synchronize()
        end

        if steps > 1
            timings = zeros(steps)
            for i = 2:steps
                print("Time to simple gemm ")
                timings[i] = @elapsed begin
                    C = A * B
                    CUDA.synchronize()
                end

                println(timings[i])
            end

            average_time = sum(timings) / (steps - 1)
            gflops = (2 * A_rows * A_cols * B_cols) * 1E-9 / average_time
            println("GFLOPS: ", gflops, " steps: ", steps)
        end


        print("Time to total time ")
    end

    return 0
end #main

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

    @time begin

        # Julia is column-based (like Fortran)
        print("Time to allocate A")
        @time A = CUDA.CuArray{Float16,2}(undef, A_rows, A_cols)

        print("Time to allocate B")
        @time B = CUDA.CuArray{Float16,2}(undef, B_rows, B_cols)

        print("Time to initialize C")
        @time C = CUDA.zeros(Float32, A_rows, B_cols)

        print("Time to fill A")
        @time CUDA.rand!(A)
        print("Time to fill B")
        @time CUDA.rand!(B)

        print("Time to simple gemm ")
        @time begin
            C = A * B
            CUDA.synchronize()
        end

        if steps > 1
            timings = zeros(steps)
            for i = 2:steps
                print("Time to simple gemm ")
                timings[i] = @elapsed begin
                    C = A * B
                    CUDA.synchronize()
                end

                println(timings[i])
            end

            average_time = sum(timings) / (steps - 1)
            gflops = (2 * A_rows * A_cols * B_cols) * 1E-9 / average_time
            println("GFLOPS: ", gflops, " steps: ", steps)
        end

        print("Time to total time ")
    end

    return 0
end #main

end # module
