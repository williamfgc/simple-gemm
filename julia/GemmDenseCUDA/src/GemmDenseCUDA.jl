module GemmDenseCUDA

import CUDA

BLOCK_SIZE = 32

function gemm!(
    A::CUDA.CuDeviceMatrix{Float32,1},
    B::CUDA.CuDeviceMatrix{Float32,1},
    C::CUDA.CuDeviceMatrix{Float32,1},
)

    row = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    col = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y

    sum = Float32(0.0)

    if row <= size(A, 1) && col <= size(B, 2)

        for i = 1:size(A, 2)
            @inbounds sum += A[row, i] * B[i, col]
        end
        C[row, col] = sum
    end

    return
end

function gemm16!(
    A::CUDA.CuDeviceMatrix{Float16,1},
    B::CUDA.CuDeviceMatrix{Float16,1},
    C::CUDA.CuDeviceMatrix{Float32,1},
)
    row = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    col = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y

    sum = Float32(0.0)

    if row <= size(A, 1) && col <= size(B, 2)

        for i = 1:size(A, 2)
            @inbounds sum += A[row, i] * B[i, col]
        end
        C[row, col] = sum
    end

    return
end




function gemm64!(
    A::CUDA.CuDeviceMatrix{Float64,1},
    B::CUDA.CuDeviceMatrix{Float64,1},
    C::CUDA.CuDeviceMatrix{Float64,1},
)

    row = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    col = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y

    sum = Float64(0.0)

    if row <= size(A, 1) && col <= size(B, 2)

        for i = 1:size(A, 2)
            @inbounds sum += A[row, i] * B[i, col]
        end
        C[row, col] = sum
    end

    return
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

        grid_rows::Int32 = cld(A_rows, BLOCK_SIZE)
        grid_cols::Int32 = cld(B_colds, BLOCK_SIZE)
        blocks = (grid_rows, grid_cols)
        threads = (BLOCK_SIZE, BLOCK_SIZE)

        # println("Threads: ", threads)
        # println("Blocks: ", blocks)

        print("Time to simple gemm ")
        @time begin
            CUDA.@cuda threads = threads blocks = blocks gemm!(A, B, C)
            CUDA.synchronize()
        end

        if steps > 1
            timings = zeros(steps)
            for i = 2:steps
                print("Time to simple gemm ")
                timings[i] = @elapsed begin
                    CUDA.@cuda threads = threads blocks = blocks gemm!(A, B, C)
                    CUDA.synchronize()
                end

                println(timings[i])
            end

            average_time = sum(timings) / (steps - 1)
            gflops = (2 * A_rows * A_cols * B_cols) * 1E-9 / average_time
            println(
                "GFLOPS: ",
                gflops,
                " steps: ",
                steps,
                " average_time: ",
                average_time,
            )
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

        grid_rows::Int32 = ceil((A_rows + BLOCK_SIZE - 1) / BLOCK_SIZE)
        grid_cols::Int32 = ceil((B_cols + BLOCK_SIZE - 1) / BLOCK_SIZE)
        blocks = (grid_rows, grid_cols)
        threads = (BLOCK_SIZE, BLOCK_SIZE)

        # println("Threads: ", threads)
        # println("Blocks: ", blocks)

        print("Time to simple gemm ")
        @time begin
            CUDA.@cuda threads = threads blocks = blocks gemm64!(A, B, C)
            CUDA.synchronize()
        end

        if steps > 1
            timings = zeros(steps)
            for i = 2:steps
                print("Time to simple gemm ")
                timings[i] = @elapsed begin
                    CUDA.@cuda threads = threads blocks = blocks gemm64!(A, B, C)
                    CUDA.synchronize()
                end

                println(timings[i])
            end

            average_time = sum(timings) / (steps - 1)
            gflops = (2 * A_rows * A_cols * B_cols) * 1E-9 / average_time
            println(
                "GFLOPS: ",
                gflops,
                " steps: ",
                steps,
                " average_time: ",
                average_time,
            )
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

        grid_rows::Int32 = ceil((A_rows + BLOCK_SIZE - 1) / BLOCK_SIZE)
        grid_cols::Int32 = ceil((B_cols + BLOCK_SIZE - 1) / BLOCK_SIZE)
        blocks = (grid_rows, grid_cols)
        threads = (BLOCK_SIZE, BLOCK_SIZE)

        # println("Threads: ", threads)
        # println("Blocks: ", blocks)

        print("Time to simple gemm ")
        @time begin
            CUDA.@cuda threads = threads blocks = blocks gemm16!(A, B, C)
            CUDA.synchronize()
        end

        if steps > 1
            timings = zeros(steps)
            for i = 2:steps
                print("Time to simple gemm ")
                timings[i] = @elapsed begin
                    CUDA.@cuda threads = threads blocks = blocks gemm16!(A, B, C)
                    CUDA.synchronize()
                end

                println(timings[i])
            end

            average_time = sum(timings) / (steps - 1)
            gflops = (2 * A_rows * A_cols * B_cols) * 1E-9 / average_time
            println(
                "GFLOPS: ",
                gflops,
                " steps: ",
                steps,
                " average_time: ",
                average_time,
            )
        end

        print("Time to total time ")
    end

    return 0
end #main


end #module
