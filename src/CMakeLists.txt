cmake_minimum_required (VERSION 3.8)


if(USE_CUDA)
  add_executable(FastCorrelation FastCorrelation.cpp correlationGPU.cu) 
  target_link_libraries(FastCorrelation cufft)
else()
  add_executable(FastCorrelation FastCorrelation.cpp) 
endif()


find_package(FFTW QUIET REQUIRED COMPONENTS SIMPLE)

if(NOT FFTW_FOUND OR USE_MKL)
  find_package(FFTW QUIET REQUIRED COMPONENTS MKL SIMPLE)
  set(USE_MKL ON)
endif()

if(FFTW3_INCLUDE_DIRS)
  include_directories(FastCorrelation ${FFTW3_INCLUDE_DIRS})
endif()

target_link_libraries(FastCorrelation  ${FFTW_LIBRARIES})

install(TARGETS FastCorrelation  RUNTIME DESTINATION bin)
  
configure_file ( "config.h.in" ${CMAKE_CURRENT_SOURCE_DIR}/config.h  )
