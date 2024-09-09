# Decimation
Decimation is a signal processing technique that reduces the sampling rate of a discrete-time signal by removing some of its samples and applying a low-pass filter to prevent aliasing. The primary goal of decimation is to decrease the amount of data while preserving the essential characteristics of the original signal. This process involves two main steps: filtering and downsampling. First, the signal is passed through a low-pass filter to limit its bandwidth and eliminate high-frequency components that could cause aliasing when the sampling rate is reduced. The filtered signal is then downsampled by a factor \( M \), which involves keeping every \( M \)-th sample and discarding the rest. Mathematically, if \( x[n] \) is the original signal and \( h[n] \) is the impulse response of the low-pass filter, the decimated signal \( y[m] \) can be expressed as:
\[ y[m] = x[mM] \]
where \( M \) is the decimation factor. The filtering step can be represented as:
\[ z[n] = (x * h)[n] = \sum_{k=-\infty}^{\infty} x[k] \cdot h[n-k] \]
followed by downsampling:
\[ y[m] = z[mM] \]
Decimation is widely used in various applications such as digital communications, audio signal processing, and data compression. In digital communication systems, decimation is employed to reduce the data rate of signals, making them more efficient for transmission and storage. For instance, in a digital radio receiver, the received signal might be decimated to match the sampling rate required by subsequent processing stages, such as demodulation and decoding. In audio signal processing, decimation is used to convert high-sample-rate audio recordings to lower sample rates for efficient storage and playback, while preserving the audio quality. For example, a studio-quality audio signal sampled at 96 kHz may be decimated to 48 kHz for distribution, which reduces the file size and data rate while maintaining sufficient audio fidelity. Decimation is also essential in multi-rate signal processing systems, where signals are processed at different sampling rates for various tasks, such as filtering, analysis, and synthesis.


## Matlab Code

```matlab
%{
Aim: Demonstrating CUDA code for decimation
%}

%% Basic Setup
clc; clear;

%% Compiling Code
mexcuda decimate_cuda.cu;

%% Preparing Inputs
samplingFactor = 2;
samplingFrequency = 16000;
timeArray = 0:(1/samplingFrequency):1;
frequency0 = 20; frequency1 = 6000;
inputArray = sin(2*pi*frequency0*timeArray) + sin(2*pi*frequency1*timeArray);
inputArray = inputArray(:);

% design filter
filter_order = 11;
filter_cutoff = 0.5;
filter_window = hamming(filter_order+1);
filterKernel = fir1(filter_order, filter_cutoff, 'low', filter_window);
filterKernel = filterKernel(:);

%% Calling Function
outputArray = decimate_cuda(inputArray, filterKernel,samplingFactor);

%% Plotting Results
figure(1);
subplot(2,1,1); plot(inputArray); title("Input Signal");
subplot(2,1,2); plot(outputArray); title("Filtered and decimated Signal");
```


## CUDA Code

```C
// Aim: Perform decimation using cuda

// headers 
#include "mex.h"
#include "../Beamforming/booktools.h"
#include<cuda_runtime.h>

// kernel
__global__ void conv(double *d_inputPointerA,
                        double *d_inputPointerB,
                        const int samplingFactor,
                        double *d_outputPointer,
                        const mwSize *d_inputDimensionsA,
                        const mwSize *d_inputDimensionsB)
{
    // shared memory
    extern __shared__ double sharedMem[];

    // length of input-array
    const int inputLength = (int)(d_inputDimensionsA[0]); 

    // factor by which we're sampling
    // const int samplingFactor = (int)(d_inputPointerC[0]); 

    // the index for which we're calculating the result for
    const int indexAssignedToBlock = (int)(blockIdx.x*samplingFactor); 

    // the farthest index from the index, for which this block is calculating dot product for
    int indicesOfKernelUtilized = blockIdx.x - d_inputDimensionsB[0]+1; 
    indicesOfKernelUtilized = min((int)indicesOfKernelUtilized, (int)d_inputDimensionsB[0]);

    // checking if the value for which we're calculating convolution is within the bound of the input array
    if(indexAssignedToBlock<inputLength)
    {
        // finding index assigned to this particular thread
        int indexAssignedToThisThread = indexAssignedToBlock - threadIdx.x;

        // checking if the thread assigned to this 
        if (indexAssignedToThisThread >=0 || indexAssignedToThisThread < inputLength)
        {
            // saving value to the shared memory
            sharedMem[threadIdx.x] = (double)d_inputPointerA[indexAssignedToThisThread]*
                                        (double)d_inputPointerB[threadIdx.x];
        }
        else{
            // if it is not within the scope, assign a zero to the index
            sharedMem[threadIdx.x] = (double)0;
        }

        // syncing the threads within the block
        __syncthreads();

        // assigning the first thread to take care of the addition
        if(threadIdx.x == 0)
        {       
            // the variable to hold the accumulation results
            double accusum = 0;

            // Summing up the values stored in the shared memory
            for(int i = 0; i<indicesOfKernelUtilized; ++i)
            {
                accusum = accusum + (double)sharedMem[i];
            }

            // copying the shared-memory into the value
            d_outputPointer[blockIdx.x] = accusum;
        }
    }
}

// mex-function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // check number of inputs
    if(nrhs!=3)
    {
        mexErrMsgTxt("The Number of Inputs are Wrong \n");
    }
    
    // check number of outputs
    if(nlhs!=1)
    {
        mexErrMsgTxt("Number of expected outputs are wrong \n");
    }
    
    // check data-types
    /*
        first argument: input array
        second argument: input kernel
        third argument: downsampling factor
    */ 
    if(!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
    {
        mexErrMsgTxt("First argument has the wrong data-type \n");
    }
    if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) )
    {
        mexErrMsgTxt("Second argument has the wrong data type \n");
    }
    if(!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) )
    {
        mexErrMsgTxt("Third argument has the wrong data type \n");
    }

    // Fetching the Input Data
    CustomGPUObject inputArray(prhs[0]);
    CustomGPUObject kernelArray(prhs[1]);
    const int samplingFactor = (int)mxGetScalar(prhs[2]);

    // Transferring data to GPU
    inputArray.copyFromHostToDevice();
    kernelArray.copyFromHostToDevice();
    
    // setup output
    const int inputLength = (int)inputArray.inputDimensions[0];
    const int kernelLength = (int)kernelArray.inputDimensions[0];
    const int outputLength = (int)((inputLength+samplingFactor-1)/samplingFactor);
    plhs[0] = mxCreateNumericMatrix(outputLength, 
                                    1,
                                    mxDOUBLE_CLASS, 
                                    mxREAL);
    CustomGPUObject outputArray(plhs[0]);
    
    // Function Calls
    dim3 blockConfiguration(kernelLength, 1, 1);
    dim3 gridConfiguration(outputLength,1,1);   
    conv<<<gridConfiguration,
            blockConfiguration,
            kernelLength*sizeof(double)>>>(inputArray.d_inputPointer_real,
                                            kernelArray.d_inputPointer_real,
                                            samplingFactor,
                                            outputArray.d_inputPointer_real,
                                            inputArray.d_inputDimensions,
                                            kernelArray.d_inputDimensions);

    // Copying results from device to host
    outputArray.copyFromDeviceToHost();

    // shutting down
    cudaDeviceSynchronize();
    cudaDeviceReset();

}
```