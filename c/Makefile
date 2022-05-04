
CC=clang
CC_FLAGS=-O3 -march=native -fopenmp
# Wombat
# CC_FLAGS=-O3 -march=native -fopenmp -mcpu=neoverse-v1

ifneq (,$(findstring clang,$(CC)))
all: clang
else
all: gemm_dense_common
endif

.PHONY: gemm_dense_common

gemm_dense_common: gemm-dense-openmp.c
	$(CC) gemm-dense-openmp.c -S -o gemm-dense-openmp-$(CC).s $(CC_FLAGS)
	$(CC) gemm-dense-openmp.c -o gemm-dense-openmp-$(CC) $(CC_FLAGS)
	
clang: gemm_dense_common
	$(CC) gemm-dense-openmp.c -S -emit-llvm $(CC_FLAGS)


.PHONY: clean

clean : 
	rm *.s *.ll *-gcc *-*clang