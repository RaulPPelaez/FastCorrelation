/*Raul P. Pelaez 2018. Computes the correlation using cuFFT.
  See FastCorrelation.cu.

 */
#ifndef CORRELATIONGPU_CUH
#define CORRELATIONGPU_CUH


#include"common.h"
#include<cufft.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include"superRead.h"
#include"cufftPrecisionAgnostic.h"

namespace FastCorrelation{

  template<class real>
  __global__ void convolution(cufftComplex_t<real> *A, cufftComplex_t<real> *B,
			      cufftComplex_t<real> *output, int N, real prefactor){
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=N) return;

    const auto a = A[i];
    const auto b = B[i];
    output[i].x =(a.x*b.x+a.y*b.y);
    output[i].y = (a.y*b.x - a.x*b.y);
  }

  template<class real>
  void correlationGPUFFT(FILE *in,
			 int numberElements,
			 int nsignals,
			 int windowSize,
			 int maxLag,
			 bool padSignal,
			 ScaleMode scaleMode){    
    using cufftComplex_t = cufftComplex_t<real>;
    using cufftReal_t = cufftReal_t<real>;

    //I interpret time windows as just more signals, and store them as such.
    //So a single signal with 2 time windows is taken as 2 signals with half the time of the original signal.  
    int numberElementsReal = numberElements;
    int nsignalsReal = nsignals;
    int nWindows = numberElements/windowSize;
    numberElements = windowSize;
    nsignals *= nWindows;

    //Each signal is duplicated and the second part filled with zeros, this way the results are equivalent to the usual O(N^2) correlation algorithm.
    int numberElementsPadded = numberElements;
    if(padSignal)
      numberElementsPadded = numberElements + maxLag;
    thrust::device_vector<cufftComplex_t> d_signalA;//(numberElementsPadded*nsignals);
    thrust::device_vector<cufftComplex_t> d_signalB;//(numberElementsPadded*nsignals);

    thrust::host_vector<cufftComplex_t> h_signalA(numberElementsPadded*nsignals, cufftComplex_t());
    thrust::host_vector<cufftComplex_t> h_signalB(numberElementsPadded*nsignals, cufftComplex_t());

    {
      cufftReal_t* h_signalA_ptr = (cufftReal_t*)thrust::raw_pointer_cast(h_signalA.data());
      cufftReal_t* h_signalB_ptr = (cufftReal_t*)thrust::raw_pointer_cast(h_signalB.data());

      std::vector<double> means(nsignalsReal*2, 0);
      //Read input
      for(int i = 0; i<numberElementsReal*nsignalsReal; i++){
	double tmp[2];
	readNextLine(in, 2, tmp);
	//This is a little obtuse but this codes the input format into the windows=signals convention. Each window
	// provides nsignalsReal additional signals, which are placed after the originals for each time.
	//The global offset for the current time window (the index of the first signal of time 0 in current window)
	const int windowOffset = (i/(nsignalsReal*numberElements))*nsignalsReal;
	//The current signal index (from 0 to nSignalsReal)
	const int signalStride = i%nsignalsReal;
	//The index of the first signal at the current time
	const int timeStride = ((i%(nsignalsReal*numberElements))/nsignalsReal)*nWindows*nsignalsReal;
	const int index = timeStride + signalStride + windowOffset;
	h_signalA_ptr[index] = tmp[0];
	h_signalB_ptr[index] = tmp[1];
	means[2*signalStride] += tmp[0];
	means[2*signalStride+1] += tmp[1];     
      }
      for(int signalStride = 0; signalStride<nsignalsReal; signalStride++){
	means[2*signalStride]   /= numberElementsReal;
	means[2*signalStride+1] /= numberElementsReal;
      }
      for(int i = 0; i<numberElementsReal*nsignalsReal; i++){
	const int windowOffset = (i/(nsignalsReal*numberElements))*nsignalsReal;
	const int signalStride = i%nsignalsReal;
	const int timeStride = ((i%(nsignalsReal*numberElements))/nsignalsReal)*nWindows*nsignalsReal;
	const int index = timeStride + signalStride + windowOffset;
	h_signalA_ptr[index] -= means[2*signalStride];
	h_signalB_ptr[index] -= means[2*signalStride+1];
      }
    }
    //From this point the data can be used as nsignals of size numberElementsPadded
    
    //Upload
    d_signalA = h_signalA; cufftComplex_t *d_signalA_ptr = thrust::raw_pointer_cast(d_signalA.data());
    d_signalB = h_signalB; cufftComplex_t *d_signalB_ptr = thrust::raw_pointer_cast(d_signalB.data());
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);


    //FFT both signals
    cufftHandle plan;
    {
      //Special plan for interleaved signals
      int rank = 1;           // --- 1D FFTs
      int n[] = { numberElementsPadded };   // --- Size of the Fourier transform
      int istride = nsignals, ostride = nsignals;        // --- Distance between two successive input/output elements
      int idist = 1, odist = 1; // --- Distance between batches
      int inembed[] = { 0 };    // --- Input size with pitch (ignored for 1D transforms)
      int onembed[] = { 0 };    // --- Output size with pitch (ignored for 1D transforms)
      int batch = nsignals;     // --- Number of batched executions
      cufftPlanMany(&plan, rank, n, 
		    inembed, istride, idist,
		    onembed, ostride, odist, CUFFT_Real2Complex<real>::value, batch);
    }
    cufftSetStream(plan, stream);
    cufftExecReal2Complex<real>(plan, (cufftReal_t*) d_signalA_ptr, d_signalA_ptr);
    cufftExecReal2Complex<real>(plan, (cufftReal_t*) d_signalB_ptr, d_signalB_ptr);

  
    //Convolve TODO: there are more blocks than necessary
    int Nthreads=512;
    int Nblocks =(nsignals*(numberElementsPadded+1))/Nthreads+1;
    convolution<real><<<Nblocks, Nthreads, 0, stream>>>(d_signalA_ptr,
					     d_signalB_ptr,
					     d_signalB_ptr, //Overwrite this array
					     nsignals*(numberElementsPadded+1),
					     1/(double(numberElements)));

    //Inverse FFT the convolution to obtain correlation
    cufftHandle plan2;
    {
      //Special plan for interleaved signals
      int rank = 1;                           // --- 1D FFTs
      int n[] = { numberElementsPadded };                 // --- Size of the Fourier transform
      int istride = nsignals, ostride = nsignals;        // --- Distance between two successive input/output elements
      int idist = 1, odist = 1; // --- Distance between batches
      int inembed[] = { 0 };          // --- Input size with pitch (ignored for 1D transforms)
      int onembed[] = { 0 };         // --- Output size with pitch (ignored for 1D transforms)
      int batch = nsignals;                      // --- Number of batched executions
      cufftPlanMany(&plan2, rank, n, 
		    inembed, istride, idist,
		    onembed, ostride, odist, CUFFT_Complex2Real<real>::value, batch);
    }

    cufftSetStream(plan2, stream);
    cufftExecComplex2Real<real>(plan2, d_signalB_ptr, (cufftReal_t*)d_signalB_ptr);
    //Download
    h_signalA = d_signalB; //signalB now holds correlation

    
    cufftReal_t* h_signalA_ptr = (cufftReal_t*)thrust::raw_pointer_cast(h_signalA.data());

    //Write results
    for(int i = 0; i<maxLag; i++){
      double mean = 0.0;
      double mean2 = 0.0;
      
      double scale;
      switch(scaleMode){
      case ScaleMode::biased:   scale = 1; break;
      case ScaleMode::unbiased: scale = double(maxLag)/(maxLag-i+1); break;
      case ScaleMode::none:     scale = maxLag; break;
      default: scale = 1; break;
      }
      //I really do not know where this comes from, but it works... I think I messed some normalization in the FFT
      double misteriousScale =0.5*(2-double(maxLag)/numberElements);
      

      double normFFT = double(numberElements)*numberElements;
      for(int s=0; s<nsignals;s++){ //Average all correlations
	double tmp = h_signalA_ptr[nsignals*i+s];
	mean += tmp;
	mean2 += tmp*tmp;
      }
      mean  /= double(nsignals)*normFFT;
      mean2 /= double(nsignals)*normFFT*normFFT;
      
      double corr = mean*misteriousScale*scale;
      double error = sqrt(mean2 - mean*mean)*misteriousScale*scale;
      std::cout<<i<<" "<<corr<<" "<<error<<std::endl;

    }

    cufftDestroy(plan);
    cufftDestroy(plan2);
    cudaStreamDestroy(stream);
  }
}
#endif