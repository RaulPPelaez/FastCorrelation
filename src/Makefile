

#Uncomment this line to use boost for file reading, this improves performance up to a ~20%
#USE_BOOST=-DUSE_BOOST

all:
	nvcc -arch=sm_52  -std=c++11 -O3 FastCorrelation.cu -lcufft $(USE_BOOST) -o correlation

clean:
	rm -f correlation
