/*Raul P. Pelaez 2018. FastCorrelation
  Computes the correlation of two signals (autocorrelation if the signals are the same).
  
  For usage, format and instructions run with: correlation -h

COMPILATION:

Modify src/Makefile if necesary. You can tune there which library is used for reading and if the code is compiled in CPU only or hybrid GPU/CPU mode.

Compile with make



 */



#include"parseArguments.h"
#include"common.h"
#include"config.h"
#include<iostream>
#include"correlationGPU.h"
#include"correlationCPU.h"

using namespace FastCorrelation;

enum class Device{GPU=0,CPU};
enum class Precision{single_prec=0, double_prec};

//Calls the appropiate version of FastCorrelation (either CPU or GPU with either float or double precision)
template<class ... T>
void dispatchFastCorrelation(Precision prec, Device device, T... args){  
  if(device == Device::GPU){
    if(prec==Precision::single_prec) GPU::correlationFFT<float>(args...);
    else if(prec==Precision::double_prec) GPU::correlationFFT<double>(args...);
  }
  else if(device == Device::CPU){
    if(prec==Precision::single_prec) CPU::correlationFFT<float>(args...);
    else if(prec==Precision::double_prec) CPU::correlationFFT<double>(args...);
  }
}


void print_help();
int main(int argc, char *argv[]){

  //Parse command line options
  if(checkFlag(argc, argv, "-h") or checkFlag(argc, argv, "-help")){print_help(); exit(1);}
  
  int numberElements;   if(!parseArgument(argc, argv, "-N", &numberElements)){print_help();exit(1);}
  int nsignals = 1;     parseArgument(argc, argv, "-nsignals", &nsignals);
  int windowSize = numberElements;    parseArgument(argc, argv, "-windowSize", &windowSize);
  int maxLag = windowSize;   if(parseArgument(argc, argv, "-maxLag", &maxLag)){
    if(maxLag>windowSize){
      std::cerr<<"WARNING!: You should not ask for a lag time larger than the window size!"<<std::endl;
    }
  }
  bool padSignal = !checkFlag(argc, argv, "-noPad");  
  ScaleMode scaleMode = ScaleMode::biased;
  {//Scale mode
    std::string scale="biased"; parseArgument(argc, argv, "-scale", &scale);
    if(scale.compare("biased") == 0)        scaleMode = ScaleMode::biased;
    else if(scale.compare("unbiased") == 0) scaleMode = ScaleMode::unbiased;
    else if(scale.compare("none") == 0)     scaleMode = ScaleMode::none;
    else{ std::cerr<<"ERROR: Unrecognized scale mode!"<<std::endl; print_help(); exit(1);}
  }
  
  //Precision mode
  std::string prec="float"; parseArgument(argc, argv, "-prec", &prec);
  std::string dev="CPU"; parseArgument(argc, argv, "-device", &dev);

  Device device;
  Precision precision;
  if(prec.compare("float") == 0) precision = Precision::single_prec;
  else if(prec.compare("double") == 0) precision = Precision::double_prec;
  else{
    std::cerr<<"ERROR: unsupported precision!"<<std::endl;
    print_help();
    exit(1);
  }


  if(dev.compare("CPU") == 0) device = Device::CPU;
  else if(dev.compare("GPU") == 0) device = Device::GPU;
  else{
    std::cerr<<"ERROR: unsupported device! "<<dev<<std::endl;
    print_help();
    exit(1);
  }
  
  //Compute correlation
  dispatchFastCorrelation(precision, device,
			  stdin,
			  numberElements,
			  nsignals,
			  windowSize,
			  maxLag,
			  padSignal,
			  scaleMode);
 

  return 0;
}

void print_help(){
  using std::cerr;
  cerr<<"Raul P. Pelaez 2018. FastCorrelation"<<std::endl;
  cerr<<"  Computes the correlation of two signals (autocorrelation if the signals are the same)."<<std::endl;
  cerr<<"  "<<std::endl;
  cerr<<"  USAGE:"<<std::endl;
  cerr<<"     $ cat two_column_file  | correlation -N [N]  > corr.dat"<<std::endl;
  cerr<<"     "<<std::endl;
  cerr<<"     The output will have three columns:"<<std::endl;
  cerr<<"	lag corr(lag) std(lag)"<<std::endl;
  cerr<<""<<std::endl;
  cerr<<"     std = sqrt(<corr^2> - <corr>^2)  "<<std::endl;
  cerr<<""<<std::endl;
  cerr<<""<<std::endl;
  cerr<<"    If you have several files with realizations of the same signal, say signal1, signal2,...signalN"<<std::endl;
  cerr<<"      you can interleave them to use with correlation as:"<<std::endl;
  cerr<<"     $ paste -d'\\n' signal* | correlation ... > corr.dat"<<std::endl;
  cerr<<""<<std::endl;
  cerr<<"  INPUT FORMAT:"<<std::endl;
  cerr<<"     The input must have two columns with the following order:"<<std::endl;
  cerr<<"       A_t0_signal1 B_t0_signal1"<<std::endl;
  cerr<<"       A_t0_signal2 B_t0_signal2"<<std::endl;
  cerr<<"       ..."<<std::endl;
  cerr<<"       A_t0_signal_nsignals   B_t0_signal_nsignals"<<std::endl;
  cerr<<"       A_t1_signal_nsignals   B_t1_signal_nsignals"<<std::endl;
  cerr<<"       ..."<<std::endl;
  cerr<<"       A_t_N_signal_nsignals  B_t_N_signal_nsignals"<<std::endl;
  cerr<<"       "<<std::endl;
  cerr<<"     "<<std::endl;
  cerr<<"  OPTIONS:"<<std::endl;
  cerr<<"    -N: signal length"<<std::endl;
  cerr<<"    -nsignals: number of signals in the file (they all must have length N) (default is 1)."<<std::endl;
  cerr<<"    -windowSize: If present the signal will be cut in pieces of windowSize size for averaging (default is N)."<<std::endl;
  cerr<<"    -maxLag: Compute up to lag maxLag (default is N). A lower value will increase performance and the results will be numerically identical up to maxLag."<<std::endl;
  cerr<<"    -noPad: If not present the signall will be padded with zeros up to maxLag in FFT mode. (default is false)"<<std::endl;
  cerr<<"    -scale: Scale mode (default is biased), can be:"<<std::endl;
  cerr<<"          none:   Return the unscaled correlation, R."<<std::endl;
  cerr<<"          biased: Return the biased average, R/N"<<std::endl;
  cerr<<"	  unbiased: Return the unbiased average, R(t)/(N-t)"<<std::endl;
  cerr<<"    -prec: float or double, specify the precision mode (default is float)."<<std::endl;
  cerr<<"    -device: GPU or CPU, specify the device (default is CPU) to run FFT."<<std::endl;
  cerr<<""<<std::endl;
  cerr<<"    -h or -help: Print this info."<<std::endl;

}
