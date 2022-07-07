
using Test
import CUDA
import GemmDenseCUDA


@testset "test-gemm" begin

    function test_gemm(A_rows, A_cols, B_cols)
        B_rows = A_cols
        A = CUDA.CuArray{Float32,2}(undef, A_rows, A_cols)
        B = CUDA.CuArray{Float32,2}(undef, B_rows, B_cols)
        C = CUDA.zeros(Float32, A_rows, B_cols)
        CUDA.rand!(A)
        CUDA.rand!(B)

        max_threads = 16
        threads_x = min(max_threads, size(C, 1))
        threads_y = min(max_threads ÷ threads_x, size(C, 2))
        threads = (threads_x, threads_y)
        blocks = ceil.(Int, (size(C, 1), size(C, 2)) ./ threads)
        CUDA.@cuda threads = threads blocks = blocks GemmDenseCUDA.gemm(A, B, C)
        CUDA.synchronize()

        C_expected = Array(A) * Array(B)
        @test isapprox(C_expected, Array(C))
        return
    end

    test_gemm(5, 5, 5)
    test_gemm(5, 10, 5)
    test_gemm(2, 4, 6)
    test_gemm(10, 10, 10)

end

@testset "test-main" begin

    @test CUDA.@time GemmDenseCUDA.main(["5", "5", "5"]) == 0
    @test CUDA.@time GemmDenseCUDA.main(["10", "10", "10"]) == 0
    @test CUDA.@time GemmDenseCUDA.main(["100", "100", "100"]) == 0
    @test CUDA.@time GemmDenseCUDA.main(["1000", "1000", "1000"]) == 0
    @test CUDA.@time GemmDenseCUDA.main(["2000", "2000", "2000"]) == 0
    @test CUDA.@time GemmDenseCUDA.main(["5000", "5000", "5000"]) == 0
    #@test @time GemmDenseCUDA.main(["10000", "10000", "10000"]) == 0

end