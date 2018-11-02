cmake_minimum_required (VERSION 3.8)
include(CheckLanguage)
#set( CMAKE_VERBOSE_MAKEFILE on )
project (FastCorrelation CXX)
set (FastCorrelation_VERSION_MAJOR 1)
set (FastCorrelation_VERSION_MINOR 0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_BOOST "Use Boost-qi for reading, highly improves reading performance" OFF)
option(USE_MKL "Uses MKL FFTW interface instead of FFTW3" OFF)
option(DONT_USE_CUDA "Dont compile in hybrid CPU/GPU mode, requires nvcc" OFF)

if(NOT DONT_USE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(USE_CUDA ON)
    
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
    
    execute_process(COMMAND bash  ${PROJECT_SOURCE_DIR}/cmake/listArchs.sh OUTPUT_VARIABLE ARCHS)
    set(CMAKE_CUDA_FLAGS ${ARCHS} ${CMAKE_CUDA_FLAGS}) 
  endif()
endif()

#set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC" ${CMAKE_CXX_FLAGS})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)




add_subdirectory(src)

