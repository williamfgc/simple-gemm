
CC=clang
CC_FLAGS=-O3 -Wall


ifeq ($(CC), clang)
all: clang
else ifeq ($(CC), gcc )
all: gcc
endif

.PHONY: gemm_dense_common

gemm_dense_common: gemm-dense-openmp.c
	$(CC) gemm-dense-openmp.c -S -o gemm-dense-openmp-$(CC).s $(CC_FLAGS) -fopenmp
	$(CC) gemm-dense-openmp.c -o gemm-dense-openmp-$(CC) $(CC_FLAGS) -fopenmp

gcc: gemm_dense_common
	
clang: gemm_dense_common
	$(CC) gemm-dense.c -S -emit-llvm $(CC_FLAGS)
	$(CC) gemm-dense-openmp.c -S -emit-llvm $(CC_FLAGS) -fopenmp


.PHONY: clean

clean : 
	rm *.s *.ll *-gcc *-clang