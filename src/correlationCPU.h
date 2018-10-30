/*Raul P. Pelaez 2018. CPU version of fast correlation. Uses FFTW for the FFT.


 */
#ifndef CORRELATIONCPU_H
#define CORRELATIONCPU_H

#include"common.h"
#include"superRead.h"
#include<fftw3.h>
#include<vector>
#include<cmath>
namespace FastCorrelation{
  namespace CPU{

    namespace detail{
      template<class real>
      struct fftw_prec_types;
      template<> struct fftw_prec_types<double>{using type = fftw_complex;};
      template<> struct fftw_prec_types<float>{using type = fftwf_complex;};

      template<class real>
      struct fftw_plan_prec;
      template<> struct fftw_plan_prec<double>{using type = fftw_plan;};
      template<> struct fftw_plan_prec<float>{using type = fftwf_plan;};
      
      

      
      template<class real> typename fftw_prec_types<real>::type* fftw_alloc_complex_prec(int N);
      template<> typename fftw_prec_types<double>::type* fftw_alloc_complex_prec<double>(int N){
#ifdef USE_MKL
	return (fftw_prec_types<double>::type*) malloc(N*sizeof(fftw_prec_types<double>::type));
#else	
	return fftw_alloc_complex(N);
#endif
      }
      template<> typename fftw_prec_types<float>::type* fftw_alloc_complex_prec<float>(int N){
#ifdef USE_MKL
	return (fftw_prec_types<float>::type*) malloc(N*sizeof(fftw_prec_types<double>::type));
#else	
	return fftwf_alloc_complex(N);
#endif
      }



      template<class real>struct fftw_plan_many_dft_r2c_prec;
      template<>struct fftw_plan_many_dft_r2c_prec<double>{
	template <class ...T> typename fftw_plan_prec<double>::type operator()(T...args){return fftw_plan_many_dft_r2c(args...);}};
      template<>struct fftw_plan_many_dft_r2c_prec<float>{
	template <class ...T> typename fftw_plan_prec<float>::type operator()(T...args){return fftwf_plan_many_dft_r2c(args...);}};

      template<class real>struct fftw_plan_many_dft_c2r_prec;
      template<>struct fftw_plan_many_dft_c2r_prec<double>{
	template <class ...T> typename fftw_plan_prec<double>::type operator()(T...args){return fftw_plan_many_dft_c2r(args...);}};
      template<>struct fftw_plan_many_dft_c2r_prec<float>{
	template <class ...T> typename fftw_plan_prec<float>::type operator()(T...args){return fftwf_plan_many_dft_c2r(args...);}};

      void fftw_execute( fftw_plan_prec<double>::type plan){::fftw_execute(plan);}
      void fftw_execute( fftw_plan_prec<float>::type plan){::fftwf_execute(plan);}

      template<class ...T>void fftw_execute_dft_r2c(fftw_plan plan, T...args){::fftw_execute_dft_r2c(plan, args...);}
      template<class ...T>void fftw_execute_dft_r2c(fftwf_plan plan, T...args){fftwf_execute_dft_r2c(plan, args...);}

      template<class ...T>void fftw_execute_dft_c2r(fftw_plan plan, T...args){::fftw_execute_dft_c2r(args...);}
      template<class ...T>void fftw_execute_dft_c2r(fftwf_plan plan, T...args){fftwf_execute_dft_c2r(args...);}
	      
    }

    template<class real>
    void correlationFFT(FILE *in,
			int numberElements,
			int nsignals,
			int windowSize,
			int maxLag,
			bool padSignal,
			ScaleMode scaleMode){
      using fftw_complex = typename detail::fftw_prec_types<real>::type;
      using fftw_plan = typename detail::fftw_plan_prec<real>::type;
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

      fftw_complex* signalA = detail::fftw_alloc_complex_prec<real>(numberElementsPadded*nsignals*2);
      fftw_complex* signalB = detail::fftw_alloc_complex_prec<real>(numberElementsPadded*nsignals*2);
      {
	real* h_signalA_ptr = (real*)signalA;
	real* h_signalB_ptr = (real*)signalB;
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

      
      //FFT both signals
      fftw_plan plan;
      {
	//Special plan for interleaved signals
	int rank = 1;           // --- 1D FFTs
	const int n[] = { numberElementsPadded };   // --- Size of the Fourier transform
	int istride = nsignals, ostride = nsignals;        // --- Distance between two successive input/output elements
	int idist = 1, odist = 1; // --- Distance between batches
	const int *inembed = NULL;//{ numberElementsPadded };    // --- Physical size = logical size
	const int *onembed = NULL;//{ numberElementsPadded };    // --- 
	int batch = nsignals;     // --- Number of batched executions
	real*  ins = (real*) signalA;
	fftw_complex*  out = (fftw_complex*) signalA;
	plan = detail::fftw_plan_many_dft_r2c_prec<real>()(rank, n, batch,
						   ins, inembed, istride, idist,
						   out, onembed, ostride, odist,
						   FFTW_ESTIMATE);
    }

      detail::fftw_execute(plan);
      detail::fftw_execute_dft_r2c(plan, (real*) signalB, (fftw_complex*) signalB);


      //Convolve
      for(int i = 0; i<nsignals*numberElementsPadded; i++){
	const fftw_complex a = {signalA[i][0], signalA[i][1]};
	const fftw_complex b = {signalB[i][0], signalB[i][1]};
	signalB[i][0] = (a[0]*b[0] + a[1]*b[1]);
	signalB[i][1] = (a[1]*b[0] - a[0]*b[1]);
      }

      //Inverse FFT the convolution to obtain correlation
      fftw_plan plan2;
      {
	//Special plan for interleaved signals
	int rank = 1;           // --- 1D FFTs
	int n[] = { numberElementsPadded };   // --- Size of the Fourier transform
	int istride = nsignals, ostride = nsignals;        // --- Distance between two successive input/output elements
	int idist = 1, odist = 1; // --- Distance between batches
	int *inembed = NULL;//{ 0 };    // --- Physical size = logical size
	int *onembed = NULL;//{ 0 };    // --- 
	int batch = nsignals;     // --- Number of batched executions
	fftw_complex*  in = (fftw_complex*) signalB;
	real*  out = (real*) signalA;
	
	plan2 = detail::fftw_plan_many_dft_c2r_prec<real>()(rank, n, batch,
							 in, inembed, istride, idist,
							 out, onembed, ostride, odist,
							 FFTW_ESTIMATE);
      }
      detail::fftw_execute(plan2);

      real* h_signalA_ptr = (real*)signalA;

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


      
    }
  }
}

#endif
