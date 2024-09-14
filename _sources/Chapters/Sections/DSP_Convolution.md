# Standard Convolution

1D convolution is a mathematical operation applied to signals, typically an input signal and a kernel(also called, ``filter''), to produce a third signal that represents how the input-signal is modified by the filter. Born from signal processing, it is widely used in neural networks, data analysis and so on. Mathematically, the 1D convolution of an input signal, $x[n]$, with a kernel, $h[k]$, is defined as 

$$
y[n] = (x * h)[n] = \sum_{k=-\infty}^{\infty} x[k] \cdot h[n-k]
$$

Here, $y[n]$ is the resulting output signal, $x[k]$ is the input signal, and $h[n-k]$ is the kernel flipped and shifted by $n$. The sum is taken over all possible values of $k$, indicating that each point in the output signal is a weighted sum of the input signal values, where the weights are given by the kernel.

In practical applications, both the input signal and the kernel are typically finite-length sequences. For a finite-length input signal \( x[n] \) of length \( N \) and a kernel \( h[k] \) of length \( M \), the convolution operation is often simplified and restricted to a finite sum. The equation for the finite-length 1D convolution is:

$$
y[n] = \sum_{k=0}^{M-1} x[n-k] \cdot h[k]
$$

for \( n = 0, 1, 2, \ldots, N + M - 2 \). In this context, appropriate handling of boundary conditions, such as zero-padding the input signal, is necessary to compute the convolution at the edges. This operation produces an output signal \( y[n] \) that is typically longer than the original input signal, with a length of \( N + M - 1 \). 1D convolution is fundamental in many signal processing techniques, including filtering, smoothing, and feature extraction, and serves as a basis for more complex operations in higher dimensions.

## Matlab Code

Here, we generate a signal that consists of a low-frequency signal and a high-frequency signal. This looks like a low-frequency signal that is noisy. The kernel we choose is a flat kernel, which means that it removes high-frequency components. The result is a smoothened version of the input flat-kernel is a low-pass filter. 


```matlab
%{
=================================================================================
Aim:
    Implementing 1D convolution for audio-signal.
=================================================================================
%}

%% Basic Setup
clc; clear; close all; 

%% Compiling mex-code
mexcuda conv00.cu

%% Preparing input signal

% global parameters
samplingFrequency = 16000;
timeArray = (1:10000)/samplingFrequency;

% first signal parameter
signalFrequencyA = 50;
inputArrayA = sin(2*pi*signalFrequencyA*timeArray);

% second signal
signalFrequencyB = 7000;
inputArrayB = sin(2*pi*signalFrequencyB*timeArray);

% adding up the two signals
inputArray = inputArrayA + inputArrayB;

% transposing the signal
inputArray = transpose(inputArray);

%% Setting up the convolution  kernel
convKernel = transpose(ones(size(1:10)));
convKernel = convKernel/sum(convKernel);

%% Calling the function
outputArray = conv00(inputArray, convKernel);

%% Plotting before vs after
figure(1);
subplot(1,2,1); plot(inputArray); title("Array: Pre-filtering \n");
subplot(1,2,2); plot(outputArray); title("Array: Post-Filtering \n");
```








## CUDA Code

The nature of convolution involves element-wise product of values and then summing them. Due to this collaborative nature of work in addition to the general rule that kernels are not too large, using shared-memory is a good idea. Thus, we allocate shared memory to each block of size corresponding to that of the kernel-length.

So each block takes care of finding the value at each index. The threads under the block find the element-wise products and copies into the corresponding linear index of the shared memory. Once all the threads have done that job, we assign the first thread in each block to sum up the values of those stored in the shared memory. The same first thread, index = 0, is then tasked with copying the dot product over to the global memory at the index, which the block is responsible for.  

```C
/*
Aim:
    Implementing function to convolve input-array with the input-kernel
*/ 

// headers 
#include "mex.h"
#include "../Beamforming/booktools.h"

// kernel
__global__ void conv(double *d_inputPointerA,
                        double *d_inputPointerB,
                        double *d_outputPointer,
                        const mwSize *d_inputDimensionsA,
                        const mwSize *d_inputDimensionsB)
{
    // shared memory
    extern __shared__ double sharedMem[];

    // miscellaneous
    int finalOutputLength = (int)d_inputDimensionsA[0] + (int)d_inputDimensionsB[0] - 1;
    int indicesOfKernelUtilized = blockIdx.x - d_inputDimensionsB[0]+1;
    indicesOfKernelUtilized = min((int)indicesOfKernelUtilized, (int)d_inputDimensionsB[0]);
    double tempOutput;

    // checking if we're inside the 
    if(blockIdx.x<finalOutputLength)
    {
        // finding index assigned to this particular thread
        int indexAssignedToThisThread = blockIdx.x - threadIdx.x;

        // checking if the thread assigned to this 
        if (indexAssignedToThisThread>=0 || indexAssignedToThisThread<finalOutputLength)
        {
            // producing the product
            tempOutput = d_inputPointerA[indexAssignedToThisThread]*d_inputPointerB[threadIdx.x];

            // saving value to the shared memory
            sharedMem[threadIdx.x] = tempOutput;
        }

        // syncing the threads within the block
        __syncthreads();

        // assigning the first thread to take care of the addition
        double accusum = 0;
        if(threadIdx.x == 0)
        {       
            // Summing up the shared-memory
            for(int i = 0; i<indicesOfKernelUtilized; ++i)
            {
                accusum = accusum + (double)sharedMem[i];
            }

            // copying the shared-memory into the value
            d_outputPointer[blockIdx.x] = accusum;
            // d_outputPointer[blockIdx.x] = blockIdx.x;

        }
    }
}

// mex-function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // check number of inputs
    if(nrhs!=2)
    {
        mexErrMsgTxt("The Number of Inputs are Wrong \n");
    }
    
    // check number of outputs
    if(nlhs!=1)
    {
        mexErrMsgTxt("Number of expected outputs are wrong \n");
    }
    
    // check data-types
    if(!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
    {
        mexErrMsgTxt("First argument has the wrong data-type \n");
    }
    if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) )
    {
        mexErrMsgTxt("Second argument has the wrong data type \n");
    }

    // Fetching the data
    CustomGPUObject inputArray(prhs[0]);
    CustomGPUObject kernelArray(prhs[1]);

    // Sending the Data to the GPU
    inputArray.copyFromHostToDevice();
    kernelArray.copyFromHostToDevice();
    
    // setup output
    size_t outputLength = inputArray.inputDimensions[0] + kernelArray.inputDimensions[0] - 1;
    plhs[0] = mxCreateNumericMatrix(outputLength,1, mxDOUBLE_CLASS, mxREAL);
    CustomGPUObject outputArray(plhs[0]); 
    outputArray.copyFromHostToDevice(); 
    
    // Preparing and calling kernel
    dim3 blockConfiguration(kernelArray.inputDimensions[0], 1, 1);
    dim3 gridConfiguration(outputLength,1,1);
    conv<<<gridConfiguration, 
            blockConfiguration,
            kernelArray.inputDimensions[0]*sizeof(double)>>>
                (inputArray.d_inputPointer_real,
                    kernelArray.d_inputPointer_real,
                    outputArray.d_inputPointer_real,
                    inputArray.d_inputDimensions,
                    kernelArray.d_inputDimensions);

    // Fetching Data from GPU to Host
    outputArray.copyFromDeviceToHost();
    
    // Shutting down
    cudaDeviceSynchronize();
    cudaDeviceReset();
}
```
