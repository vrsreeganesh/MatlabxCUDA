# Interpolation

Interpolation is a signal processing technique used to increase the resolution of a discrete signal by estimating intermediate sample values. Unlike upsampling, which merely inserts zeros between existing samples, interpolation employs mathematical algorithms to compute new sample values that smoothly bridge the gaps between original samples. This process enhances the signal's temporal or spatial resolution without introducing abrupt changes or discontinuities. A common method for interpolation is linear interpolation, where the new sample values are computed as weighted averages of neighboring original samples. Mathematically, for a signal \( x[n] \) and an interpolation factor \( L \), the interpolated signal \( y[m] \) can be represented as:
\[ y[m] = x\left[\left\lfloor \frac{m}{L} \right\rfloor\right] + \left( \frac{m}{L} - \left\lfloor \frac{m}{L} \right\rfloor \right) \cdot \left( x\left[\left\lfloor \frac{m}{L} \right\rfloor + 1\right] - x\left[\left\lfloor \frac{m}{L} \right\rfloor\right] \right) \]
where \( \lfloor \cdot \rfloor \) denotes the floor function, and \( L \) is the interpolation factor.
Interpolation is extensively used in various applications such as image processing, audio signal processing, and numerical analysis. In image processing, interpolation is crucial for resizing and resampling images, allowing for zooming in on specific regions or converting images to different resolutions. For example, bicubic interpolation is often used to resize digital photographs while maintaining image quality. In audio signal processing, interpolation is used to convert audio signals to higher sample rates for playback on different devices or to align signals in multi-track recordings. For instance, digital audio workstations (DAWs) use interpolation to ensure smooth playback and editing of audio tracks at varying sample rates. In numerical analysis, interpolation methods like spline interpolation are employed to approximate complex functions and datasets, facilitating tasks such as curve fitting, data visualization, and solving differential equations. Interpolation, by providing a means to estimate intermediate values, enables smoother transitions and more accurate representations of underlying continuous signals.


##  MATLAB Code


```matlab
%{
    Aim: Demonstrating CUDA Implementation of Signal Interpolation
%}

%% Basic Setup
clc; clear; close all;

%% Compiling
mexcuda interpolate_cuda.cu

%% Preparing inputs
interpolatingFactor = 2;
samplingFrequency = 16000;
frequency0 = 100;
timeArray = 0:(1/samplingFrequency):1e-1;
inputArray = sin(2*pi*frequency0*timeArray);

%% design filter
filter_order = 11;
filter_cutoff = (1/interpolatingFactor);
filter_window = hamming(filter_order+1);
filterKernel = fir1(filter_order, filter_cutoff, 'low', filter_window);
filterKernel = interpolatingFactor*filterKernel(:);

%% calling the function
outputArray = interpolate_cuda(inputArray(:), filterKernel(:), interpolatingFactor(:));

%% plotting
figure(1);
subplot(2,1,1); plot(inputArray); title("Input Signal");
subplot(2,1,2); plot(outputArray); title("Interpolated Signal");
```





## CUDA Code

```C
// Aim: Implementing Interpolation using Custom CUDA-kernels

// headers
#include "mex.h"
#include "../Beamforming/booktools.h"
#include<cuda_runtime.h>

// upsampling kernel
__global__ void upsample_cuda(double *d_inputPointerA,
                                const int samplingFactor,
                                double *d_outputPointer,
                                mwSize *d_inputDimensionsA)
{
    // getting global address
    const int tidx = threadIdx.x + blockIdx.x*blockDim.x;

    // miscellaneous
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

// filtering kernel
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

        }
    }
}

// gateway function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // checking number of inputs
    if(nrhs!=3)
    {
        mexErrMsgTxt("Number of Inputs are Wrong \n");
    }
    
    // checking number of outputs
    if(nlhs!=1)
    {
        mexErrMsgTxt("Number of Expected Outputs are Wrong \n");
    }

    // checking input data type
    if(!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
    {
        mexErrMsgTxt("The first argument is expected to be of type double \n");
    }
    if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]))
    {
        mexErrMsgTxt("The second argument is expected to be of type, double \n");
    }
    if(!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]))
    {
        mexErrMsgTxt("The third argument is expected to be of type, double \n");
    }

    // Fetching Inputs
    CustomGPUObject inputArray(prhs[0]);
    CustomGPUObject kernelArray(prhs[1]);
    const int samplingFactor = (int)mxGetScalar(prhs[2]);

    // Sending Inputs from Host to Device
    inputArray.copyFromHostToDevice();
    kernelArray.copyFromHostToDevice();

    // building output matrix
    const int inputLength = inputArray.inputDimensions[0];
    const int outputLength = inputLength*samplingFactor;
    plhs[0] = mxCreateNumericMatrix(outputLength,
                                    1,
                                    mxDOUBLE_CLASS,
                                    mxREAL);
    CustomGPUObject outputArray(plhs[0]);
    outputArray.copyFromHostToDevice();

    // calling the upsampling kernel
    dim3 blockConfiguration_upsampling(32,1,1); 
    dim3 gridConfiguration_upsampling(\
        (int)((inputLength+blockConfiguration_upsampling.x-1)/blockConfiguration_upsampling.x),\
        1,\
        1);
    upsample_cuda<<<gridConfiguration_upsampling,
                    blockConfiguration_upsampling>>>
                        (inputArray.d_inputPointer_real,
                            samplingFactor,
                            outputArray.d_inputPointer_real,
                            inputArray.d_inputDimensions);

    // updating the input values
    inputArray.inputDimensions[0] = samplingFactor*inputLength;
    cudaMemcpy(inputArray.d_inputDimensions,
                inputArray.inputDimensions,
                inputArray.numDimensions*sizeof(mwSize),
                cudaMemcpyHostToDevice);

    // copying the values from one to the other
    double *d_inputPointerAA; 
    cudaMalloc((void**)&d_inputPointerAA, outputLength*sizeof(double));
    cudaMemcpy(d_inputPointerAA,
                outputArray.d_inputPointer_real,
                outputLength*sizeof(double),
                cudaMemcpyDeviceToDevice);

    // call convolution kernel
    const int kernelLength = kernelArray.inputDimensions[0];
    dim3 blockConfiguration_filtering(kernelLength, 1, 1); 
    dim3 gridConfiguration_filtering(outputLength,1,1);
    conv<<<gridConfiguration_filtering,
            blockConfiguration_filtering,
            kernelArray.inputDimensions[0]*sizeof(double)>>>
                (d_inputPointerAA,
                    kernelArray.d_inputPointer_real,
                    outputArray.d_inputPointer_real,
                    inputArray.d_inputDimensions,
                    kernelArray.d_inputDimensions);

    // copying data from device to host
    outputArray.copyFromDeviceToHost();

    // shutting down
    cudaDeviceSynchronize();
    cudaDeviceReset();
}
```