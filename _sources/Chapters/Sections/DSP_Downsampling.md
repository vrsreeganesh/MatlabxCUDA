# Signal Downsampling

Downsampling is a signal processing technique used to reduce the sampling rate of a signal by retaining only a subset of the original samples. This process effectively reduces the data rate and the size of the dataset, making it more manageable for storage or further processing. The primary goal of downsampling is to capture the essential features of the signal while discarding redundant information. Mathematically, if $x[n]$ represents the original signal, the downsampled signal $y[m]$ can be obtained by taking every $M$-th sample of $x[n]$:

$$
y[m] = x[mM]
$$

where $M$ is the downsampling factor. This operation reduces the sampling rate by a factor of $M$, resulting in a new signal that is $1/M$ the length of the original signal. For instance, if $x[n]$ is sampled at 10 kHz and $M = 2$, the downsampled signal $y[m]$ will have a sampling rate of 5 kHz.

Downsampling is commonly used in various applications such as digital image processing, audio signal processing, and data compression. In image processing, downsampling can be used to reduce the resolution of an image, making it easier to analyze or transmit. For example, an image captured at a resolution of 4000x3000 pixels can be downsampled to 2000x1500 pixels, reducing the amount of data while preserving the overall structure of the image. In audio processing, downsampling is used to decrease the file size of audio recordings, making them more suitable for storage or streaming. For instance, an audio file sampled at 44.1 kHz can be downsampled to 22.05 kHz, which reduces the file size by half while still retaining sufficient quality for many applications. Proper filtering is usually required before downsampling to prevent aliasing, which ensures that the downsampled signal accurately represents the original signal without introducing distortions.


## Matlab Code

Here, we first compile the mexcuda code. Note that we don't need any special libraries because the kernel we're using is a custom one. This code compiles the code, then prepare the argument and then calls the function. 

```matlab
%{
Aim:
    Demonstrating CUDA implementation of down-sampling. 
%}

%% Basic setup
clc; clear; close all;

%% Compiling input code
mexcuda downsample_cuda.cu

%% preparing inputs
inputArray = sin(2*pi*1e2*linspace(0,1,2048));
inputArray = inputArray(:);
samplingFactor = 10;

%% Calling the functions
outputArray = downsample_cuda(inputArray, samplingFactor);

%% displaying the results
figure(1);
subplot(2,1,1); plot(inputArray); title("Input Array");
subplot(2,1,2); plot(outputArray); title("Output Array");
```

## CUDA Code

Here, we see the downsampling implementation of CUDA. Since this is not a collaborative task, we can implement this without the use of any specialised memory. As usual, our cuda code first does the dimensionality check, data-type check, get dimensions, pointers to the data, send the data to the device, call the function, copy the results back and shut down. 


```C
/*
Aim:
    Building custom-kernel to down-sample code. 
*/ 

// header-files
#include "mex.h"
#include "../Beamforming/booktools.h"

// global kernel
__global__ void downsample_cuda(double *d_inputPointerA,
                                const int samplingFactor,
                                double *d_outputPointer,
                                const mwSize *d_inputDimensionsA)
{
    // getting global address
    const int tidx = threadIdx.x + blockIdx.x*blockDim.x;

    // miscellaneous
    // const int samplingFactor =  (int)d_inputPointerB[0];
    const int responsibleIndex = tidx*samplingFactor;
    const int inputLength = (int)d_inputDimensionsA[0];

    /*
    If the index from which we're fetching values are within
    the length of the array, then we copy it
    */ 
    if(responsibleIndex<inputLength)
    {
        // copying the values
        d_outputPointer[tidx] = (double)d_inputPointerA[responsibleIndex];
    }

    // making sure everything is done 
    __syncthreads();
}

// gate-way function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // checking the number of inputs
    if(nrhs!=2)
    {
        mexErrMsgTxt("Number of inputs are wrong \n");
    }   

    // checking the expected number of outputs
    if(nlhs!=1)
    {
        mexErrMsgTxt("Number of expected outputs are wrong \n");
    }

    // checking the input data types
    if(!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
    {
        mexErrMsgTxt("The input array is expected to be a matrix of type, double \n");
    }
    if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || mxGetM(prhs[1])*mxGetN(prhs[1]) !=1)
    {
        mexErrMsgTxt("The second input is expected to be a scalar \n");
    }

    // Fetching Inputs 
    CustomGPUObject inputArray(prhs[0]);
    const int samplingFactor = (int)mxGetScalar(prhs[1]);

    // Transferring Assets to the Device
    inputArray.copyFromHostToDevice();

    // fetching the pointer to the input data
    double *inputPointerA = (double *)mxGetPr(prhs[0]);
    double *inputPointerB = (double *)mxGetPr(prhs[1]);
    
    // setting up output
    const int inputLength = (int)(inputArray.numElements); 
    const int outputLength = (int)((inputLength+samplingFactor-1)/samplingFactor);
    plhs[0] = mxCreateNumericMatrix(outputLength,
                                    1, 
                                    mxDOUBLE_CLASS, 
                                    mxREAL);
    CustomGPUObject outputArray(plhs[0]);
    outputArray.copyFromHostToDevice();

    // preparing for function call 
    dim3 blockConfiguration(32,1,1);
    const int numblocks = (int)((outputLength+blockConfiguration.x-1)/blockConfiguration.x);
    dim3 gridConfiguration(numblocks,1,1);
    downsample_cuda<<<gridConfiguration, 
                        blockConfiguration>>>(inputArray.d_inputPointer_real,
                                            samplingFactor,
                                            outputArray.d_inputPointer_real,
                                            inputArray.d_inputDimensions);

    // Copying data from Device to Host
    outputArray.copyFromDeviceToHost();

    // Shutting Down
    cudaDeviceSynchronize();
    cudaDeviceReset();
}
```

## References
- "Downsampling (Signal Processing)." Wikipedia, The Free Encyclopedia, 12 Sept. 2024, en.wikipedia.org/wiki/Downsampling_(signal_processing). Accessed 12 Sept. 2024.