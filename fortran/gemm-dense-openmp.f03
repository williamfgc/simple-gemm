
! Uses Modern Fortran 2003/2008 where possible, Fortran 90/95 for the most part

module GemmDenseOpenMP
use omp_lib
implicit none

contains

subroutine gemm (A, B, C, ierr)
    real(kind=4), dimension(:,:), intent(in) :: A, B
    real(kind=4), dimension(:,:), intent(inout) :: C
    integer(kind=4), intent(out) :: ierr
    ! local variables
    integer(kind=4) :: A_rows, A_cols, B_cols
    integer(kind=4) :: i, l, j
    real(kind=4) :: temp

    A_rows = size(A,1)
    A_cols = size(A,2)
    B_cols = size(B,2)

    write(*,*) A_rows, ' ', A_cols, ' ', B_cols

    !$omp parallel do default(none) &
    !$omp shared(A,B,C, B_cols, A_cols, A_rows) private(j,l,i, temp)
    do j=1,B_cols
        do l=1,A_cols
            temp = B(l,j)
            do i=1,A_rows
                C(i,j) = C(i,j) + temp * A(i,l)
            end do
        end do
    end do
    !$omp end parallel do

    write(*,*) 'Loop is done'

    ierr = 0

end subroutine

subroutine print_matrix(A, ierr)
    real(kind=4), dimension(:,:), intent(in) :: A
    integer(kind=4), intent(out) :: ierr
    ! local variables
    integer(kind=4) :: A_rows, A_cols
    integer(kind=4) :: i, j
    
    A_rows = size(A,1)
    A_cols = size(A,2)

    write(*,'(A2)', advance="no") '[ '
    do i=1,A_rows
        do j=1,A_cols
            write(*,'(F8.5,A2)', advance="no") A(i,j), ', '
        end do
    end do
    write(*,'(A2)') ' ]'

    ierr = 0 

end subroutine 

end module

program main 
    use GemmDenseOpenMP
    implicit none

    character(15) :: arg_temp
    integer(kind=4) :: A_rows, A_cols, B_rows, B_cols
    real(kind=4), dimension(:,:), allocatable:: A, B, C
    integer(kind=4) :: ierr

    ! Fortran 2003 standard
    if ( command_argument_count() /= 3 ) then 
        write(6,*) "Usage: 3 arguments: matrix A rows, matrix A cols and matrix B cols"
        stop
    end if

    call get_command_argument(1, arg_temp)
    read(arg_temp, *) A_rows
    call get_command_argument(2, arg_temp)
    read(arg_temp, *) A_cols
    B_rows = A_cols
    call get_command_argument(3, arg_temp)
    read(arg_temp, *) B_cols

    write(*,'(3(A2,I7),A2)') '[ ', A_rows, ', ', A_cols, ', ', B_cols, ' ]'

    allocate( A(A_rows, A_cols)  )
    allocate( B(B_rows, B_cols)  )
    allocate( C(A_rows, B_cols)  )

    call random_number(A)
    call random_number(B)
    ! zero initialize C
    C = 0.0

    call gemm(A,B,C, ierr)

    !call print_matrix(C, ierr)
    deallocate(A)
    deallocate(B)
    deallocate(C)

end program main

