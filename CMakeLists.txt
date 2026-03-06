cmake_minimum_required(VERSION 3.28.3)
project(cutesseract LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

include_directories(cu_tesseract/include)

add_executable(cutesseract
        cu_tesseract/src/matrix.cu
        cu_tesseract/src/main.cu)

target_link_libraries(cutesseract curand)
