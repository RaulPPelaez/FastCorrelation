

#Uncomment this line to use boost for file reading, this improves performance up to a ~20%
#USE_BOOST=-DUSE_BOOST


#The target CUDA compute capability(s), if not set it will be autodetected
#ARCH ?= 30 35 50 52 60 61
ARCH=

#CXX=clang++-5.0
CXX=g++

ifeq ($(ARCH),)
GENCODE_FLAGS:=$(shell printf '\#include<cstdio>\n int main(){int nD;cudaGetDeviceCount(&nD);for(int i=0;i<nD;i++){cudaDeviceProp dp;cudaGetDeviceProperties(&dp, i);std::printf("%%d\\n", dp.major*10+dp.minor);} return 0;}' | nvcc -Wno-deprecated-gpu-targets -x cu - -o /tmp/listarch --run | sort -g -k1 | uniq | awk 'END{system("rm -f /tmp/listarch")}{print "-gencode arch=compute_"$$1",code=sm_"$$1}')
else
$(foreach sm,$(ARCH),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif


all:
	nvcc $(GENCODE_FLAGS)  -std=c++11 -O3 FastCorrelation.cu -lcufft $(USE_BOOST) -o correlation

clean:
	rm -f correlation
