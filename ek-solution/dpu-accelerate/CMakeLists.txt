cmake_minimum_required(VERSION 3.29)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.6/)
project(Hackthon LANGUAGES CXX CUDA)
set(PROJECT_NAME "hackthon_demo")
enable_language(CUDA)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUTLASS_NVCC_ARCHS '70;75;80')

find_package(CUDAToolkit REQUIRED)

find_path(CUTLASS_INCLUDE_DIR cutlass/cutlass.h HINTS cutlass/include)
find_path(CUTLASS_UTILS_INCLUDE_DIR cutlass/util/host_tensor.h HINTS cutlass/tools/util/include)


set(CUTLASS_ROOT_DIR /root/nvhackthon/cutlass) 
set(CUTLASS_LIBRARY_KERNELS all)
set(CUTLASS_LIBRARY_KERNELS s*gemm)
set(CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

add_executable(
    ${PROJECT_NAME}
    main.cu
    misc.cu
    ffn.cu
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE 
    cuda
    cublas
) 

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
)

