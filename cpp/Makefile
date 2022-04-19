
CC=clang++
CC_FLAGS=-O3 -std=c++11 -Wall


ifeq ($(CC), clang++)
all: clang
else ifeq ($(CC), g++ )
all: gcc
endif

.PHONY: gemm_dense_common

gemm_dense_common: gemm-dense-openmp.cpp
	$(CC) gemm-dense-openmp.cpp -S -o gemm-dense-openmp-$(CC).s $(CC_FLAGS) -fopenmp
	$(CC) gemm-dense-openmp.cpp -o gemm-dense-openmp-$(CC) $(CC_FLAGS) -fopenmp

gcc: gemm_dense_common
	
clang: gemm_dense_common
	$(CC) gemm-dense-openmp.cpp -S -emit-llvm $(CC_FLAGS) -fopenmp


.PHONY: clean

clean : 
	rm *.s *.ll *-g++ *-clang++