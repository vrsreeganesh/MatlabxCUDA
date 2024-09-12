# Signal Upsampling

Upsampling is a signal processing technique used to increase the sampling rate of a discrete signal by inserting additional samples between the original ones. This process effectively increases the data rate and prepares the signal for further processing or analysis at a higher resolution. The primary goal of upsampling is to enhance the signal's temporal or spatial resolution. Mathematically, if $x[n]$ represents the original signal, the upsampled signal $y[m]$ can be obtained by inserting $ L-1$ zeros between each sample of $x[n]$:

$$
y[m] = 
\begin{cases} 
x[m/L] & \text{if } m = kL \text{ for some integer } k \\
0 & \text{otherwise}
\end{cases}
$$


where $L$ is the upsampling factor. This operation increases the sampling rate by a factor of $L$, resulting in a new signal that has $L$ times the length of the original signal. For instance, if $x[n]$ is sampled at 10 kHz and $L = 2$, the upsampled signal $ y[m]$ will have a sampling rate of 20 kHz.

Upsampling is commonly used in various applications such as digital communication, image processing, and audio signal processing. In digital communication, upsampling is used to match the sampling rate of a signal to the requirements of the transmission medium or the receiver. For example, a signal sampled at a lower rate might be upsampled to a higher rate to meet the specifications of a digital-to-analog converter (DAC). In image processing, upsampling can be used to increase the resolution of an image, making it suitable for high-resolution displays or further analysis. For instance, an image with a resolution of 1000x1000 pixels can be upsampled to 2000x2000 pixels, providing a finer grid for more detailed visualization. In audio processing, upsampling is used to improve the quality of audio playback by increasing the sampling rate, allowing for a smoother and more accurate representation of the sound. Proper filtering, such as using a low-pass filter, is usually required after upsampling to remove the spectral replicas introduced by the zero-insertion process and to ensure that the upsampled signal accurately represents the original signal.



## Matlab Code

```matlab
%% Aim: Demonstrating CUDA implementation of upsampling. 

%% Basic setup
clc; clear; close all;

%% Compiling input code
mexcuda upsample_cuda.cu;

%% preparing inputs
inputArray = 1:100;
inputArray = inputArray(:);
samplingFactor = 7;

%% Calling the functions
outputArray = upsample_cuda(inputArray, samplingFactor);

%% displaying the results
figure(1);
subplot(1,2,1); plot(inputArray); title("Input Array");
subplot(1,2,2); plot(outputArray); title("Output Array");
```


## CUDA Code

Here, we see the downsampling implementation of CUDA. Since this is not a collaborative task, we can implement this without the use of any specialised memory. As usual, our cuda code first does the dimensionality check, data-type check, get dimensions, pointers to the data, send the data to the device, call the function, copy the results back and shut down. 

```C
// Aim: Building custom-kernel to up-sample code. 

// header-files
#include "mex.h"
#include "../Beamforming/booktools.h"

// global kernel
__global__ void upsample_cuda(double *d_inputPointerA,
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

    // checking if tidx maps to a valid index of the input array
    if(tidx<inputLength)
    {
        // copying the values
        d_outputPointer[responsibleIndex] = (double)d_inputPointerA[tidx];
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

    // Sending Data to GPU
    inputArray.copyFromHostToDevice();

    // fetching the pointer to the input data
    double *inputPointerA = (double *)mxGetPr(prhs[0]);
    double *inputPointerB = (double *)mxGetPr(prhs[1]);

    // setting up output
    const int inputLength = (int)(inputArray.numElements); 
    const int outputLength = (int)(inputLength*samplingFactor);
    plhs[0] = mxCreateNumericMatrix(outputLength, 1, mxDOUBLE_CLASS, mxREAL);
    CustomGPUObject outputArray(plhs[0]);
    outputArray.copyFromHostToDevice();

    // function call
    dim3 blockConfiguration(32,1,1);
    const int numblocks = (int)((inputLength+blockConfiguration.x-1)/blockConfiguration.x);
    dim3 gridConfiguration(numblocks,1,1);
    upsample_cuda<<<gridConfiguration,
                    blockConfiguration>>>(inputArray.d_inputPointer_real,
                                            samplingFactor,
                                            outputArray.d_inputPointer_real,
                                            inputArray.d_inputDimensions);

    // Copying results from device to host
    outputArray.copyFromDeviceToHost();

    // Shutting down
    cudaDeviceSynchronize();
    cudaDeviceReset();
}
```



## References
- "Upsampling." Wikipedia, The Free Encyclopedia, 11 Sept. 2024, en.wikipedia.org/wiki/Upsampling. Accessed 12 Sept. 2024.
