
CC=clang++
CC_FLAGS=-O3 -std=c++11 -Wall


ifeq ($(CC), clang++)
all: clang
else ifeq ($(CC), g++ )
all: gcc
endif

.PHONY: gemm_dense_common

gemm_dense_common: gemm-dense.cpp
	$(CC) gemm-dense.cpp -S -o gemm-dense-$(CC).s $(CC_FLAGS)
	$(CC) gemm-dense.cpp -o gemm-dense-$(CC) $(CC_FLAGS)

gcc: gemm_dense_common
	
clang: gemm_dense_common
	$(CC) gemm-dense.cpp -S -emit-llvm $(CC_FLAGS)


.PHONY: clean

clean : 
	rm gemm-dense-* gemm-dense.ll