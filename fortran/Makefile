
FC=gfortran
FC_FLAGS=-O3 -Wall -march=native
# Wombat
# FC_FLAGS=-O3 -Wall -mcpu=neoverse-v1

ifeq ($(FC),ifort)
OMP_FLAG=-openmp
else
OMP_FLAG=-fopenmp
endif

ifneq (,$(findstring clang,$(CC)))
all: clang
else
all: gemm_dense_common
endif

.PHONY: gemm_dense_common

gemm_dense_common: gemm-dense-openmp.f03
	$(FC) gemm-dense-openmp.f03 -S -o gemm-dense-openmp-$(FC).s $(FC_FLAGS) $(OMP_FLAG)
	$(FC) gemm-dense-openmp.f03 -o gemm-dense-openmp-$(FC) $(FC_FLAGS) $(OMP_FLAG)

clang: gemm_dense_common
	$(FC) gemm-dense-openmp.f03 -S -emit-llvm $(FC_FLAGS) $(OMP_FLAG)


.PHONY: clean

clean : 
	rm *.s *.ll *-ifort *-gfortran *.mod *.tmp *-*clang
