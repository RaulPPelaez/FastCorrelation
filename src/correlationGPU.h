/*Raul P. Pelaez 2018. Computes the correlation using cuFFT.
  See FastCorrelation.cu.

 */
#ifndef CORRELATIONGPU_H
#define CORRELATIONGPU_H


#include"common.h"
#include"config.h"
#include<stdio.h>
#include<iostream>
#ifdef USE_CUDA
namespace FastCorrelation{
  namespace GPU{

    template<class real>
    void correlationFFT(FILE *in,
			int numberElements,
			int nsignals,
			int windowSize,
			int maxLag,
			bool padSignal,
			ScaleMode scaleMode);
  }
}
#else
namespace FastCorrelation{
  namespace GPU{
    template<class real>
    void correlationFFT(FILE *in,
			   int numberElements,
			   int nsignals,
			   int windowSize,
			   int maxLag,
			   bool padSignal,
			   ScaleMode scaleMode){
      std::cerr<<"ERROR: This binary was not compiled with GPU mode activated, you cant use the -GPU flag"<<std::endl;
    }

  }
}
#endif


#endif
