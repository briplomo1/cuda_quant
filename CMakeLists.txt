cmake_minimum_required(VERSION 3.24)
project(CudaProject LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES native) # compile for the server's GPU (Blackwell)

add_executable(calib cuda/calib.cu)
