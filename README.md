applies a wavelet scatter transform to a .wav file and seriates the coefficients to reconstruct them into an output .wav

the input filename and parameters have to be edited manually at the moment.  J is the number of octaves in the wavelet scatter and and Q is the number of wavelets per octave.  increasing either of them increases both the resolution of the coefficients and the computational demand of the script.

the chunking function is there to save ram.  i dont know how much ram this script uses but my 32gb are not sufficient to run this at any parameter setting.  i have a feeling it is an extremely demanding function.

if it says killed try reducing the parameters to make it morr ram efficient.  if it says it needs a 2d input that means the chunks are too small to be analyzed and the variable needs to be increased.

optionally, the percentage variable can be lowered to filter out less significant features from the scatter transform.
