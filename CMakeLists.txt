cmake_minimum_required(VERSION 3.18)
project(CudaProject LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(calib cuda/calib.cu)
