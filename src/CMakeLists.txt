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
  include_directories(FastCorrelation ${FFTW_fftw3.h_DIRS})
  set(USE_MKL ON)
endif()

include_directories(FastCorrelation ${FFTW_INCLUDE_DIRS})
link_directories(FastCorrelation ${FFTW_LIBRARY_DIRS})

target_link_libraries(FastCorrelation  ${FFTW_LIBRARIES})

install(TARGETS FastCorrelation  RUNTIME DESTINATION bin)
  
configure_file ( "config.h.in" ${CMAKE_CURRENT_SOURCE_DIR}/config.h  )
