
CC=clang++
CC_FLAGS=-O3 -std=c++11 -Wall


ifeq ($(CC), clang++)
all: clang
else ifeq ($(CC), g++ )
all: gcc
endif

.PHONY: gemm_dense_common

gemm_dense_common: gemm-dense-thread.cpp
	$(CC) gemm-dense-thread.cpp -S -o gemm-dense-thread-$(CC).s $(CC_FLAGS) -pthread
	$(CC) gemm-dense-thread.cpp -o gemm-dense-thread-$(CC) $(CC_FLAGS) -pthread

gcc: gemm_dense_common
	
clang: gemm_dense_common
	$(CC) gemm-dense-thread.cpp -S -emit-llvm $(CC_FLAGS) -pthread


.PHONY: clean

clean : 
	rm *.s *.ll *-g++ *-clang++