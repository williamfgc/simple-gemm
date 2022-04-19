
using Test
import CUDA
import GemmDenseCUDA


@testset "test-gemm" begin

    function test_gemm(A_rows, A_cols, B_cols)::Bool
        B_rows = A_cols
        A = CUDA.CuArray{Float32,2}(undef, A_rows, A_cols)
        B = CUDA.CuArray{Float32,2}(undef, B_rows, B_cols)
        C = CUDA.zeros(Float32, A_rows, B_cols)
        CUDA.rand!(A)
        CUDA.rand!(B)
        C = A * B
        C_h = Array(C)
        C_expected = Array(A) * Array(B)
        return isapprox(C_expected, C_h)
    end

    @test test_gemm(5, 5, 5)
    @test test_gemm(5, 10, 5)
    @test test_gemm(2, 4, 6)
    @test test_gemm(10, 10, 10)

end

@testset "test-main" begin

    @test CUDA.@time GemmDenseCUDA.main(["5", "5", "5"]) == 0
    @test CUDA.@time GemmDenseCUDA.main(["10", "10", "10"]) == 0
    @test CUDA.@time GemmDenseCUDA.main(["100", "100", "100"]) == 0
    @test CUDA.@time GemmDenseCUDA.main(["1000", "1000", "1000"]) == 0
    @test CUDA.@time GemmDenseCUDA.main(["2000", "2000", "2000"]) == 0
    @test CUDA.@profile GemmDenseCUDA.main(["5000", "5000", "5000"]) == 0
    #@test @time GemmDenseCUDA.main(["10000", "10000", "10000"]) == 0

end