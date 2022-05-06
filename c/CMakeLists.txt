add_executable(gemm_dense_c_omp gemm-dense-openmp.c)
target_link_libraries(gemm_dense_c_omp OpenMP::OpenMP_C)
target_compile_definitions(gemm_dense_c_omp PUBLIC _OPENMP)
