cmake_minimum_required (VERSION 2.6)



configure_file ( "config.h.in" "config.h"  )

#set(CMAKE_CUDA_STANDARD 11)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)


option(USE_CUDA "Compile in hybrid CPU/GPU mode, requires nvcc" OFF)
option(USE_BOOST "Use Boost-qi for reading, highly improves reading performance" OFF)
option(USE_MKL "Uses MKL FFTW interface instead of FFTW3" OFF)



find_package(CUDA)
if(CUDA_FOUND)
  set(USE_CUDA ON)
  if(NOT ARCHS)

    if(CUDA_VERSION_STRING VERSION_GREATER 8.5)
      set(ARCHS "3.0 3.2 3.5 3.7 5.0 5.2 5.3 6.0 6.2")
    else()
      set(ARCHS "2.0 2.1 3.0 3.2 3.5 3.7 5.0 5.2 5.3 6.0 6.2")
    endif()
    cuda_select_nvcc_arch_flags(GENCODE ${ARCHS})
  endif()
  set(CUDA_NVCC_FLAGS "-DNDEBUG -O3 -std=c++11" ${GENCODE} ${CUDA_NVCC_FLAGS})

  cuda_compile(CUOBJ correlationGPU.cu)
  cuda_add_executable(FastCorrelation FastCorrelation.cpp ${CUOBJ})
  cuda_add_cufft_to_target(FastCorrelation)

else()
  set(USE_CUDA OFF)
  add_executable(FastCorrelation FastCorrelation.cpp)
endif()



find_package(FFTW QUIET REQUIRED COMPONENTS SIMPLE)

if(NOT FFTW_FOUND)
  find_package(FFTW QUIET REQUIRED COMPONENTS MKL SIMPLE)
  set(USE_MKL ON)
endif()

if(FFTW3_INCLUDE_DIRS)
  include_directories(FastCorrelation ${FFTW3_INCLUDE_DIRS})
endif()

target_link_libraries(FastCorrelation  ${FFTW_LIBRARIES})


install(TARGETS FastCorrelation  RUNTIME DESTINATION bin)
  
