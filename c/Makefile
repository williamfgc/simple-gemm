
CC=clang
CC_FLAGS=-O3 -march=native -fopenmp

# CUDA select compute capability of your targeted hardware
# Wombat
CUDA_ARCH=80
# CC_FLAGS=-O3 -fopenmp 
# CC_FLAGS=-O3 -fopenmp -mcpu=neoverse-v1 -> results in seg fault

# Crusher GPU
ROCBLAS_DIR=/opt/rocm-5.2.0
# CC_FLAGS=-O3 --amdgpu-target=gfx90a

ifneq (,$(findstring clang,$(CC)))
all: clang
else
all: gemm_dense_common
endif

.PHONY: gemm_dense_common

blas: gemm-dense-blas.c
	$(CC) gemm-dense-blas.c -o gemm-dense-blas-$(CC) $(CC_FLAGS)

blas64: gemm-dense-blas64.c
	$(CC) gemm-dense-blas64.c -o gemm-dense-blas64-$(CC) $(CC_FLAGS)

hip: gemm-dense-hip.cpp
	hipcc $(CC_FLAGS) gemm-dense-hip.cpp -o gemm-dense-hip

hip64: gemm-dense-hip64.cpp
	hipcc $(CC_FLAGS) gemm-dense-hip64.cpp -o gemm-dense-hip64

rocblas: gemm-dense-rocblas.cpp
	hipcc $(CC_FLAGS) gemm-dense-rocblas.cpp -o gemm-dense-rocblas -I$(ROCBLAS_DIR)/include -L$(ROCBLAS_DIR)/lib -lrocblas

rocblas64: gemm-dense-rocblas64.cpp
	hipcc $(CC_FLAGS) gemm-dense-rocblas64.cpp -o gemm-dense-rocblas64 -I$(ROCBLAS_DIR)/include -L$(ROCBLAS_DIR)/lib -lrocblas

cublas: gemm-dense-cublas.cu
	nvcc -O3 -arch=sm_$(CUDA_ARCH) -Wno-deprecated-gpu-targets gemm-dense-cublas.cu -o gemm-dense-cublas -lcublas

cublas64: gemm-dense-cublas64.cu
	nvcc -O3 -arch=sm_$(CUDA_ARCH) -Wno-deprecated-gpu-targets gemm-dense-cublas64.cu -o gemm-dense-cublas64 -lcublas

cuda: gemm-dense-cuda.cu
	nvcc -O3 -arch=sm_$(CUDA_ARCH) -Wno-deprecated-gpu-targets gemm-dense-cuda.cu -o gemm-dense-cuda

cuda64: gemm-dense-cuda.cu
	nvcc -O3 -arch=sm_$(CUDA_ARCH) -Wno-deprecated-gpu-targets gemm-dense-cuda64.cu -o gemm-dense-cuda64

openmp: gemm-dense-openmp.c
	$(CC) gemm-dense-openmp.c -o gemm-dense-openmp-$(CC) $(CC_FLAGS)

openmp64: gemm-dense-openmp.c
	$(CC) gemm-dense-openmp64.c -o gemm-dense-openmp64-$(CC) $(CC_FLAGS)

gemm_dense_common: gemm-dense-openmp.c
	$(CC) gemm-dense-openmp.c -S -o gemm-dense-openmp-$(CC).s $(CC_FLAGS)
	$(CC) gemm-dense-openmp.c -o gemm-dense-openmp-$(CC) $(CC_FLAGS)
	
clang: gemm_dense_common
	$(CC) gemm-dense-openmp.c -S -emit-llvm $(CC_FLAGS)


.PHONY: clean

clean : 
	rm *.s *.ll *-gcc *-*clang gemm-dense-cuda gemm-dense-cuda64 gemm-dense-hip gemm-dense-hip64 gemm-dense-cublas gemm-dense-cublas64