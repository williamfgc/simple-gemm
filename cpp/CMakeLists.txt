add_library(gemm_dense_cpp_common OBJECT gemm-dense-common.cpp)
target_include_directories(gemm_dense_cpp_common PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
)

if(${BLAS_FOUND})
    add_executable(gemm_dense_cpp_blas gemm-dense-blas.cpp)
    target_link_libraries(gemm_dense_cpp_blas PUBLIC
        gemm_dense_cpp_common
        ${BLAS_LIBRARIES} 
    )
endif()

add_executable(gemm_dense_cpp_threads gemm-dense-thread.cpp)
target_link_libraries(gemm_dense_cpp_threads PUBLIC
    gemm_dense_cpp_common
    Threads::Threads
)

add_executable(gemm_dense_cpp_futures gemm-dense-futures.cpp)
target_link_libraries(gemm_dense_cpp_futures PUBLIC
    gemm_dense_cpp_common
    Threads::Threads
)

add_executable(gemm_dense_cpp_exec gemm-dense-exec.cpp)
target_link_libraries(gemm_dense_cpp_exec PUBLIC
    gemm_dense_cpp_common
    TBB::tbb
)
