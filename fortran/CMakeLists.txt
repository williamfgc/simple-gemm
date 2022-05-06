add_executable(gemm_dense_f_omp gemm-dense-openmp.f90)
# target_sources(gemm_dense_f_omp PUBLIC gemm-dense-openmp.f90)
target_link_libraries(gemm_dense_f_omp OpenMP::OpenMP_Fortran)
# set_target_properties(gemm_dense_f_omp PROPERTIES LINKER_LANGUAGE Fortran)
