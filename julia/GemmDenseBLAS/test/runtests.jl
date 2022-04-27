
using Test
import Random
import GemmDenseBLAS

@testset "test-gemm" begin

    function test_gemm(A_rows, A_cols, B_cols)::Bool
        B_rows = A_cols
        A = Array{Float32,2}(undef, A_rows, A_cols)
        B = Array{Float32,2}(undef, B_rows, B_cols)
        C = zeros(Float32, A_rows, B_cols)
        Random.rand!(A)
        Random.rand!(B)
        GemmDenseBLAS.gemm!(A, B, C)
        C_expected = A * B
        return isapprox(C_expected, C)
    end

    @test test_gemm(5, 5, 5)
    @test test_gemm(5, 10, 5)
    @test test_gemm(2, 4, 6)
    @test test_gemm(10, 10, 10)

end

@testset "test-main" begin

    @test @time GemmDenseBLAS.main(["5", "5", "5"]) == 0
    @test @time GemmDenseBLAS.main(["10", "10", "10"]) == 0
    @test @time GemmDenseBLAS.main(["100", "100", "100"]) == 0
    @test @time GemmDenseBLAS.main(["1000", "1000", "1000"]) == 0
    @test @time GemmDenseBLAS.main(["10000", "10000", "10000"]) == 0

end