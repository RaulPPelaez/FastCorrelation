Raul P. Pelaez 2018. FastCorrelation  
Computes the correlation of two signals (autocorrelation if the signals are the same).  
You can compute using a GPU with cuFFT or using a CPU with FFTW  
    
##  USAGE:  
```bash
$ cat two_column_file  | correlation -N [N]  > corr.dat  
```   

The output will have three columns:
```
	lag corr(lag) std(lag)
```
std = sqrt(<corr^2> - <corr>^2)  

If you have several files with realizations of the same signal, say signal1, signal2,...signalN  you can interleave them to use with correlation as:  
```bash
$ paste -d'\n' signal* | correlation ... > corr.dat  
```
  
##  INPUT FORMAT:  
The input must have two columns with the following order:  
``` 
       A_t0_signal1 B_t0_signal1  
       A_t0_signal2 B_t0_signal2  
       ...  
       A_t0_signal_nsignals   B_t0_signal_nsignals  
       A_t1_signal_nsignals   B_t1_signal_nsignals  
       ...  
       A_t_N_signal_nsignals  B_t_N_signal_nsignals  
```         

##  OPTIONS:  
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
	-device: GPU or CPU, specify the device (default is CPU) to run FFT.  
    -h or -help: Print this info.  
