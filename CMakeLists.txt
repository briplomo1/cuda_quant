cmake_minimum_required(VERSION 3.24)
project(CudaProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES native) # compile for the server's GPU (Blackwell)

# P1: Algorithm 1 in CUDA
add_executable(calib cuda/calib.cu common/calib_cpu.cpp)
target_include_directories(calib PRIVATE common)

# P2: Algorithm 1 in MPI, built only where MPI is installed (the server)
find_package(MPI COMPONENTS CXX)
if (MPI_CXX_FOUND)
    add_executable(calib_mpi mpi/calib_mpi.cpp)
    target_include_directories(calib_mpi PRIVATE common)
    target_link_libraries(calib_mpi PRIVATE MPI::MPI_CXX)
endif ()
