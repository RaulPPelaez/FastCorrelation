/*Raul P. Pelaez 2018. FastCorrelation
  Computes the correlation of two signals (autocorrelation if the signals are the same).
  
  USAGE:
     $ cat two_column_file  | correlation -N [N]  > corr.dat
     
     The output will have three columns:
	lag corr(lag) std(lag)

     std = sqrt(<corr^2> - <corr>^2)  


    If you have several files with realizations of the same signal, say signal1, signal2,...signalN
      you can interleave them to use with correlation as:
     $ paste -d'\n' signal* | correlation ... > corr.dat

  INPUT FORMAT:
     The input must have two columns with the following order:
       A_t0_signal1 B_t0_signal1
       A_t0_signal2 B_t0_signal2
       ...
       A_t0_signal_nsignals   B_t0_signal_nsignals
       A_t1_signal_nsignals   B_t1_signal_nsignals
       ...
       A_t_N_signal_nsignals  B_t_N_signal_nsignals
       
     
  OPTIONS:
    -N: signal length
    -nsignals: number of signals in the file (they all must have length N) (default is 1).
    -windowSize: If present the signal will be cut in pieces of windowSize size for averaging (default is N).
    -maxLag: Compute up to lag maxLag (default is N). A lower value will increase performance and the results will be numerically identical up to maxLag.
    -noPad: If not present the signall will be padded with zeros up to maxLag in FFT mode. (default is false)
    -scale: Scale mode (default is biased), can be:
          none:   Return the unscaled correlation, R.
          biased: Return the biased average, R/N
	  unbiased: Return the unbiased average, R(t)/(N-t)
    -prec: float or double, specify the precision mode (default is float).

    -h or -help: Print this info.

   
 */



#include"parseArguments.h"
#include"correlationGPU.cuh"

using namespace FastCorrelation;
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
  

  //Compute correlation

  
  if(prec.compare("float") == 0){
    correlationGPUFFT<float>(stdin,
			     numberElements,
			     nsignals,
			     windowSize,
			     maxLag,
			     padSignal,
			     scaleMode);
  }
  else if(prec.compare("double") == 0){
    correlationGPUFFT<double>(stdin,
			     numberElements,
			     nsignals,
			     windowSize,
			     maxLag,
			     padSignal,
			     scaleMode);
  }
  else{
    std::cerr<<"ERROR: Unrecognized precision mode!! Select float or double!."<<std::endl;
    print_help();
    exit(1);
  }
 

  return 0;
}

void print_help(){
std::cerr<<"Raul P. Pelaez 2018. FastCorrelation"<<std::endl;
std::cerr<<"  Computes the correlation of two signals (autocorrelation if the signals are the same)."<<std::endl;
std::cerr<<"  "<<std::endl;
std::cerr<<"  USAGE:"<<std::endl;
std::cerr<<"     $ cat two_column_file  | correlation -N [N]  > corr.dat"<<std::endl;
std::cerr<<"     "<<std::endl;
std::cerr<<"     The output will have three columns:"<<std::endl;
std::cerr<<"	lag corr(lag) std(lag)"<<std::endl;
std::cerr<<""<<std::endl;
std::cerr<<"     std = sqrt(<corr^2> - <corr>^2)  "<<std::endl;
std::cerr<<""<<std::endl;
std::cerr<<""<<std::endl;
std::cerr<<"    If you have several files with realizations of the same signal, say signal1, signal2,...signalN"<<std::endl;
std::cerr<<"      you can interleave them to use with correlation as:"<<std::endl;
std::cerr<<"     $ paste -d'\n' signal* | correlation ... > corr.dat"<<std::endl;
std::cerr<<""<<std::endl;
std::cerr<<"  INPUT FORMAT:"<<std::endl;
std::cerr<<"     The input must have two columns with the following order:"<<std::endl;
std::cerr<<"       A_t0_signal1 B_t0_signal1"<<std::endl;
std::cerr<<"       A_t0_signal2 B_t0_signal2"<<std::endl;
std::cerr<<"       ..."<<std::endl;
std::cerr<<"       A_t0_signal_nsignals   B_t0_signal_nsignals"<<std::endl;
std::cerr<<"       A_t1_signal_nsignals   B_t1_signal_nsignals"<<std::endl;
std::cerr<<"       ..."<<std::endl;
std::cerr<<"       A_t_N_signal_nsignals  B_t_N_signal_nsignals"<<std::endl;
std::cerr<<"       "<<std::endl;
std::cerr<<"     "<<std::endl;
std::cerr<<"  OPTIONS:"<<std::endl;
std::cerr<<"    -N: signal length"<<std::endl;
std::cerr<<"    -nsignals: number of signals in the file (they all must have length N) (default is 1)."<<std::endl;
std::cerr<<"    -windowSize: If present the signal will be cut in pieces of windowSize size for averaging (default is N)."<<std::endl;
std::cerr<<"    -maxLag: Compute up to lag maxLag (default is N). A lower value will increase performance and the results will be numerically identical up to maxLag."<<std::endl;
std::cerr<<"    -noPad: If not present the signall will be padded with zeros up to maxLag in FFT mode. (default is false)"<<std::endl;
std::cerr<<"    -scale: Scale mode (default is biased), can be:"<<std::endl;
std::cerr<<"          none:   Return the unscaled correlation, R."<<std::endl;
std::cerr<<"          biased: Return the biased average, R/N"<<std::endl;
std::cerr<<"	  unbiased: Return the unbiased average, R(t)/(N-t)"<<std::endl;
std::cerr<<"    -prec: float or double, specify the precision mode (default is float)."<<std::endl;
std::cerr<<""<<std::endl;
std::cerr<<"    -h or -help: Print this info."<<std::endl;

}
