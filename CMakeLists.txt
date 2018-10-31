cmake_minimum_required (VERSION 2.6)
project (FastCorrelation CXX)
set (FastCorrelation_VERSION_MAJOR 1)
set (FastCorrelation_VERSION_MINOR 0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


set( CMAKE_VERBOSE_MAKEFILE on )

set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC" ${CMAKE_CXX_FLAGS})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


add_subdirectory(src)

