
FC=gfortran
FC_FLAGS=-O3 -Wall -cpp

ifeq ($(FC),ifort)
OMP_FLAG=-openmp
else ifeq ($(FC),gfortran)
OMP_FLAG=-fopenmp
endif

all: gemm-dense-openmp.f03
	$(FC) gemm-dense-openmp.f03 -S -o gemm-dense-openmp-$(FC).s $(FC_FLAGS) $(OMP_FLAG)
	$(FC) gemm-dense-openmp.f03 -o gemm-dense-openmp-$(FC) $(FC_FLAGS) $(OMP_FLAG)

.PHONY: clean

clean : 
	rm *.s *.ll *-ifort *-gfortran *.mod *.tmp
