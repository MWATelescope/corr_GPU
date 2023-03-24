# corr_GPU
A simple, fast, standalone nvidia GPU-based FX correlator suitable streamed data applications.

The code is based on [Wayth et al, 2009 - "A GPU-based Real-time Software Correlation System for the Murchison Widefield Array Prototype"](https://ui.adsabs.harvard.edu/abs/2009PASP..121..857W/abstract), developed for the MWA 32T era, and has been tweaked over time to cater for many different input data types.

The code is based on CUDA and requires a CUDA capable GPU. It takes a stream of input samples, converts the samples to floating point (real or complex), FFTs into a user-specified number of channels for each input, forms the cross correlation over all inputs (maintaining spectral information) and accumulates the results over a user-specified number of averages. 

### Dependencies
- CUDA [see here](https://developer.nvidia.com/cuda-downloads)
- Ubuntu packages: build-essential

### Building
`make` should do the trick. There is also an option to make install, by default to /usr/local/bin

Run the code with no command-line args for a usage summary.

## Data formats
corr_GPU was originally built for a streaming data environment and to have simple output files that could be easily inspected.

### Input
The input, from a file (or named pipe) or stdin, is implicitly a stream of interleaved data samples, one for each input per timestep, then the same for the next timestep etc. Input can be real or complex. There is no header or metadata in the input stream.  
E.g for a system with 4 inputs and complex data, the input stream is  
r0,i0,r1,i1,r2,i2,r3,i3,r0,i0,r1,i1,r2,i2,r3,i3... etc for successive timesteps.

The actual type of data (byte, int, float, signed or unsigned) is specified on the command-line.

### Output
The output of the code matches Frank Briggs' original "L-File" format, with separate files for the cross correlations and auto correlations. (There are pros and cons of doing it this way, but it is advantageous for debugging and commissioning.)
There is a command-line option to choose which outputs are enabled, autocorrelations, cross-correlations or both. A new output format has also been added to output ASTRON/LOFAR style "ACC" files, although with single precision as described below.

Note that the code makes no assumptions about the nature of the inputs (e.g successive inputs might be two polarisations of the same antenna) and autocorrelations are only those of an input with itself. The cross correlation of two different polarisations on the same antenna is just another cross-correlation.

LACSPC files: Autocorrelations have the file suffix ".LACSPC", and are binary files with single precision float (written in the native format of whatever machine the code is run on) with implicit dimensions [time][input][channel]. Just to be clear, time index is changing most slowly and channel index is changing most quickly.

LCCSPC files: cross-correlations have the file suffix ".LCCSPC" and are binary files with single precision float complex (i.e. real then imaginary) with implicit dimensions [time][product_index][channel] where there are N(N-1)/2 cross products for N inputs.

The order of products is determined as per the example below
```
prod_ind=0  
for (inp1=0; inp1<N; inp1++) {  
    for (inp2=inp1; inp2<N; inp2++ {  
        prod_ind++  
    }  
}
```

ACC files: this puts all output data into the same file including both autos and cross correlations. The autocorrelations in this format are still written as complex floats, so will typically have large real component and small or zero imaginary component.
The implicit dimensions of the output data are [time][product][channel] where there are now N(N+1)/2 products.
