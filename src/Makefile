

#Uncomment this line to use boost for file reading, this improves performance up to a ~20%, but it seems a little brittle between boost versions
#USE_BOOST=-DUSE_BOOST

USE_GPU=-DGPUMODE

#The target CUDA compute capability(s), if not set it will be autodetected
#ARCH ?= 30 35 50 52 60 61
ARCH=
#CUDA root folder
CUDA_ROOT=/usr/local/cuda
#CXX=clang++-5.0
CXX=g++

#You can include here the fftw3 location if it is not already included in the LD_LIBRARY_PATH and standard include paths
FFTW_FLAGS= -lfftw3 -lfftw3f

#Uncomment if you want to use the MKL wrapper interface to FFTW instead of FFTW itself
#FFTW_FLAGS=-DMKL_ILP64 -m64 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
#USE_MKL=-DUSE_MKL

ifdef USE_GPU
GPUFLAGS= $(USE_GPU) -L$(CUDA_ROOT)/lib64 GPU.o  -lcufft -lcudart
ifeq ($(ARCH),)
GENCODE_FLAGS:=$(shell printf '\#include<cstdio>\n int main(){int nD;cudaGetDeviceCount(&nD);for(int i=0;i<nD;i++){cudaDeviceProp dp;cudaGetDeviceProperties(&dp, i);std::printf("%%d\\n", dp.major*10+dp.minor);} return 0;}' | $(CUDA_ROOT)/bin/nvcc -Wno-deprecated-gpu-targets -x cu - -o /tmp/listarch --run | sort -g -k1 | uniq | awk 'END{system("rm -f /tmp/listarch")}{print "-gencode arch=compute_"$$1",code=sm_"$$1}')
else
$(foreach sm,$(ARCH),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif

endif

all: GPU.o
#-DMKL_ILP64 -m64 -I${MKLROOT}/include  -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
	$(CXX) -std=c++11 -O3 FastCorrelation.cpp  $(FFTW_FLAGS) $(GPUFLAGS) $(USE_BOOST) $(USE_MKL) $(USE_GPU) -o correlation -Wall -Wextra

GPU.o:
ifdef USE_GPU
	$(CUDA_ROOT)/bin/nvcc -std=c++11 -O3 $(GENCODE_FLAGS) $(USE_BOOST) $(USE_GPU) $(USE_MKL) -c correlationGPU.cu -o GPU.o -Wall -Wextra
endif	
clean:
	rm -f correlation GPU.o
