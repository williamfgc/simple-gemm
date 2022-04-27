
CC=clang++
CC_FLAGS=-O3 -std=c++11 -Wall -march=native -mavx2


ifeq ($(CC), clang++)
all: clang
else 
all: gemm_dense_common
endif

.PHONY: gemm_dense_common

gemm_dense_common: gemm-dense-thread.cpp
	$(CC) gemm-dense-thread.cpp -S -o gemm-dense-thread-$(CC).s $(CC_FLAGS) -pthread
	$(CC) gemm-dense-thread.cpp -o gemm-dense-thread-$(CC) $(CC_FLAGS) -pthread
	$(CC) gemm-dense-blas.cpp -S -o gemm-dense-blas-$(CC).s $(CC_FLAGS)
	$(CC) gemm-dense-blas.cpp -o gemm-dense-blas-$(CC) $(CC_FLAGS) /usr/lib/x86_64-linux-gnu/openblas-openmp/libblas.so -fopenmp

	
clang: gemm_dense_common
	$(CC) gemm-dense-thread.cpp -S -emit-llvm $(CC_FLAGS) -pthread
	$(CC) gemm-dense-blas.cpp -S -emit-llvm $(CC_FLAGS)


.PHONY: clean

clean : 
	rm *.s *.ll *-g++ *-clang++
