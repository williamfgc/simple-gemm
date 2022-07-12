
using Test
using BenchmarkTools
import Random
import GemmDenseThreads

@testset "test-gemm" begin

    function test_gemm(A_rows, A_cols, B_cols)::Bool
        B_rows = A_cols
        A = Array{Float32,2}(undef, A_rows, A_cols)
        B = Array{Float32,2}(undef, B_rows, B_cols)
        C = zeros(Float32, A_rows, B_cols)
        Random.rand!(A)
        Random.rand!(B)
        GemmDenseThreads.gemm!(A, B, C)
        C_expected = A * B
        return isapprox(C_expected, C)
    end

    function test_gemm64(A_rows, A_cols, B_cols)::Bool
        B_rows = A_cols
        A = Array{Float64,2}(undef, A_rows, A_cols)
        B = Array{Float64,2}(undef, B_rows, B_cols)
        C = zeros(Float64, A_rows, B_cols)
        Random.rand!(A)
        Random.rand!(B)
        GemmDenseThreads.gemm64!(A, B, C)
        C_expected = A * B
        return isapprox(C_expected, C)
    end

    function test_gemm16(A_rows, A_cols, B_cols)::Bool
        B_rows = A_cols
        A = Array{Float16,2}(undef, A_rows, A_cols)
        B = Array{Float16,2}(undef, B_rows, B_cols)
        C = zeros(Float32, A_rows, B_cols)
        Random.rand!(A)
        Random.rand!(B)
        GemmDenseThreads.gemm16!(A, B, C)
        C_expected = A * B
        return isapprox(C_expected, C)
    end

    @test test_gemm(5, 5, 5)
    @test test_gemm(5, 10, 5)
    @test test_gemm(2, 4, 6)
    @test test_gemm(10, 10, 10)

    @test test_gemm64(5, 5, 5)
    @test test_gemm64(5, 10, 5)
    @test test_gemm64(2, 4, 6)
    @test test_gemm64(10, 10, 10)

    @test test_gemm16(5, 5, 5)
    @test test_gemm16(5, 10, 5)
    @test test_gemm16(2, 4, 6)
    @test test_gemm16(10, 10, 10)

end

@testset "test-main" begin

    @test @time GemmDenseThreads.main(["5", "5", "5"]) == 0
    @test @time GemmDenseThreads.main(["10", "10", "10"]) == 0
    @test @time GemmDenseThreads.main(["100", "100", "100"]) == 0
    @test @time GemmDenseThreads.main(["1000", "1000", "1000"]) == 0
    @test @time GemmDenseThreads.main(["2000", "2000", "2000"]) == 0

end
