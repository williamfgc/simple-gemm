cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(simplegemm LANGUAGES C CXX Fortran)

find_package(OpenMP REQUIRED COMPONENTS C CXX Fortran)
find_package(Threads REQUIRED)
find_package(BLAS)
find_package(TBB REQUIRED)

add_subdirectory(c)
add_subdirectory(cpp)
add_subdirectory(fortran)
