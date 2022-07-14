
CC=clang
CC_FLAGS=-O3 -march=native -fopenmp
# Wombat
# CC_FLAGS=-O3 -fopenmp 
# CC_FLAGS=-O3 -fopenmp -mcpu=neoverse-v1 -> results in seg fault
# CUDA select compute capability of your targeted hardware
# Wombat CUDA_ARCH=80
CUDA_ARCH=50


ifneq (,$(findstring clang,$(CC)))
all: clang
else
all: gemm_dense_common
endif

.PHONY: gemm_dense_common

hip: gemm-dense-hip.cpp
	hipcc -O3 gemm-dense-hip.cpp -o gemm-dense-hip

hip64: gemm-dense-hip.cpp
	hipcc -O3 gemm-dense-hip64.cpp -o gemm-dense-hip64

cuda64: gemm-dense-cuda.cu
	nvcc -O3 -arch=sm_$(CUDA_ARCH) -Wno-deprecated-gpu-targets gemm-dense-cuda64.cu -o gemm-dense-cuda64

cuda: gemm-dense-cuda.cu
	nvcc -O3 -arch=sm_$(CUDA_ARCH) -Wno-deprecated-gpu-targets gemm-dense-cuda.cu -o gemm-dense-cuda

gemm_dense_common: gemm-dense-openmp.c
	$(CC) gemm-dense-openmp.c -S -o gemm-dense-openmp-$(CC).s $(CC_FLAGS)
	$(CC) gemm-dense-openmp.c -o gemm-dense-openmp-$(CC) $(CC_FLAGS)
	
clang: gemm_dense_common
	$(CC) gemm-dense-openmp.c -S -emit-llvm $(CC_FLAGS)


.PHONY: clean

clean : 
	rm *.s *.ll *-gcc *-*clang gemm-dense-cuda gemm-dense-cuda64 gemm-dense-hip gemm-dense-hip64