# Interpolation

Interpolation is a Signal Processing technique used to increase the sampling-rate of a discrete signal by estimating the intermediate sample values. Unlike upsampling, which merely inserts zeros in between existing samples, interpolation employs filtering after upsampling to compute new sample values that smoothly bridge the gap between original samples. Mathematically, it consists of the two following steps

$$
x_{up\_sampled}[i*M,j*M] = x[i,j] \\
x_{interpolated} = \mathcal{F}^{-1}(\mathcal{H}_{filter} \times \mathcal{F}_{up\_sampled}(f))
$$
where
1. $x$ is the input-signal
2. $x_{up\_sampled}$ is the upsampled input-signal
3. $\mathcal{H}_{filter}$ is the fourier transform of the filter
4. $\mathcal{F}_{up\_sampled}(f)$ is the fourier-transform of the input-signal
5. $x_{interpolated}$ is the interpolated signal

<div style="margin-top: 6mm;"></div>

Interpolation is extensively used in various applications such as image processing, audio signal processing, and numerical analysis. In audio signal processing, interpolation is used to convert audio signals to higher sample rates for playback on different devices or to align signals in multi-track recordings. For instance, digital audio workstations (DAWs) use interpolation to ensure smooth playback and editing of audio tracks at varying sample rates. In numerical analysis, interpolation methods like spline interpolation are employed to approximate complex functions and datasets, facilitating tasks such as curve fitting, data visualization, and solving differential equations. Interpolation, by providing a means to estimate intermediate values, enables smoother transitions and more accurate representations of underlying continuous signals.

<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->

##  MATLAB Code
In the Matlab code, as usual, we proceed to do the following steps
<div style="margin-top: -3mm;"></div>

1. Compile CUDA code
2. Setup the arguments
3. Function call
4. Exhibit results

```Matlab
%% Compiling
mexcuda interpolate_cuda.cu
```


<!-- ======================================================================= -->
Next, we prepare the inputs to the function. For interpolation, we need the input signal, the filter and the sampling-factor. For the input-signal, we create a simple sinusoid of frequency 100 with a sampling factor of 16KHz. For the filter, we design a low-pass FIR filter of order 11. 
```Matlab
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
filterKernel = fir1(filter_order,
                    filter_cutoff, 
                    'low', 
                    filter_window);
filterKernel = interpolatingFactor*filterKernel(:);
```
<!-- ======================================================================= -->
Next, we call the kernel, receive the output and plot the before and after.
```Matlab
%% calling the function
outputArray = interpolate_cuda(inputArray(:), filterKernel(:), interpolatingFactor(:));

%% plotting
figure(1);
subplot(2,1,1); plot(inputArray); title("Input Signal");
subplot(2,1,2); plot(outputArray); title("Interpolated Signal");
```

### Final Matlab Code
<!-- ======================================================================= -->
Putting it all together we get the following

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




<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
## CUDA Code
For the CUDA part, we define two functions: the mex-gateway function, up-sampling kernel, and the convolution kernel-definition.

### Upsampling Kernel
<!-- ======================================================================= -->
The kernel takes in the following arguments
<div style="margin-top: -3mm;"></div>

1. Pointer to the input-signal in the GPU memory-space
2. Sampling factor
3. Pointer to the memory allocated for the output, in the GPU memory-space
4. Pointer to the array containing input-array dimensions. 
```C
// upsampling kernel
__global__ void upsample_cuda(double *d_inputPointerA,
                                const int samplingFactor,
                                double *d_outputPointer,
                                mwSize *d_inputDimensionsA)
{
    ...
}
```
<!-- ======================================================================= -->
Next, we assign the threads to the data-index it is responsible for. The mapping is produced in the following manner
```C
    // getting global address
    const int tidx = threadIdx.x + blockIdx.x*blockDim.x;

    // mapping and miscellaneous
    const int responsibleIndex = tidx*samplingFactor;
    const int inputLength = (int)d_inputDimensionsA[0];
```
<!-- ======================================================================= -->
Next, we check the validity of the index assigned and based on that, we assign the values. This is upsampling, since it is the first step in interpolation. 
```C
    // checking if tidx maps to a valid index of the input array
    if(tidx<inputLength)
    {
        // copying the values
        d_outputPointer[responsibleIndex] = (double)d_inputPointerA[tidx];
    }
```

<!-- ======================================================================= -->
Putting it together, the upsampling kernel should look like this
```C
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

```

### Convolution Kernel

<!-- ======================================================================= -->
The convolution kernel expects the following arguments
<div style="margin-top: -3mm;"></div>


1. Pointer to the input-signal in the GPU global-memory
2. Pointer to the filter-array in the GPU global-memory
3. Pointer to the memory allocated to store the output, in the GPU global-memory
4. Pointer to the array holding the input-signal's dimensions, in the GPU global-memory. 
5. Pointer to the array holding the kernel-matrix's dimensions, in the GPU global-memory
```C
// filtering kernel
__global__ void conv(double *d_inputPointerA,
                     double *d_inputPointerB,
                     double *d_outputPointer,
                     const mwSize *d_inputDimensionsA,
                     const mwSize *d_inputDimensionsB)
{
    ...
}
```
<!-- ======================================================================= -->
Due to the collaborative nature of convolution, remember, we use shared-memory. And since the kernel-size is not fixed, we'll be using dynamic shared-memory which allows us to choose the shared-memory size programmatically. Hence, we declare the pointer to the shared-memory in the dynamic way, in the following manner
```C
    // shared memory
    extern __shared__ double sharedMem[];
```
<!-- ======================================================================= -->
We setup some variables that will assist us in carrying out this operation
```C
// miscellaneous
    int finalOutputLength = (int)d_inputDimensionsA[0] + \
                            (int)d_inputDimensionsB[0] - 1;
    int indicesOfKernelUtilized = blockIdx.x - d_inputDimensionsB[0]+1;
    indicesOfKernelUtilized = min((int)indicesOfKernelUtilized, (int)d_inputDimensionsB[0]);
    double tempOutput;
```

<!-- ======================================================================= -->
Next, we implement the convolution operation. In convolution, each block is assigned the responsibility of producing each output sample. Each block performs the inner-product of the kernel and the neighbour-samples according to the laws of convolution. So we find the index of the element assigned to the current threadd. Afer validity check, we calculate the element-wise product and store to the shared-memory. If the validity check fails, we store a zero to the shared-memory so that during accumulation, the results are not corrupted are garbage values. After all the element-wise products are calculated and stored to the shared-memory, we assign the first thread to accumulate the values that are stored to the shared-memory. PLease note that the __syncthreads() are required so that all the element-wise product operations are completed before the accumulation begins. The accumulated value is finally stored to the appropriate index of the output-array in the global-memory. 
```C
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
```

<!-- ======================================================================= -->
Putting it all together, the convolution kernel definition is as follows
```C
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
    int finalOutputLength = (int)d_inputDimensionsA[0] + \
                            (int)d_inputDimensionsB[0] - 1;
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
```

### Gateway Function
<!-- ======================================================================= -->
The first step in the gate-way function is to check the validity of the inputs. This involves checking the number of inputs, number of expected outputs and the data-type and dimensions of the inputs. 
```C
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

    ...
}
```

<!-- ======================================================================= -->
Once verification has been completed, we encapsulate the input-arguments with objects of the class, *CustomGPUObject*. The data is then made available in the GPU memory-space using the class method, *copyFromHostToDevice()*. Note that we do not use the class to encapulsate the sampling-factor because there is no need for explicit CUDA-API calls to make scalars available in the device memory-space. 
```C
    // Fetching Inputs
    CustomGPUObject inputArray(prhs[0]);
    CustomGPUObject kernelArray(prhs[1]);
    const int samplingFactor = (int)mxGetScalar(prhs[2]);

    // Sending Inputs from Host to Device
    inputArray.copyFromHostToDevice();
    kernelArray.copyFromHostToDevice();
```

<!-- ======================================================================= -->
Next, we create a Matlab matrix to store the output of the operation. We use the function, [*mxCreateNumericMatrix*](https://www.mathworks.com/help/matlab/apiref/mxcreatenumericmatrix.html), and the matrix created by this function is encapsulated using an object of the class, *CustomGPUObject*. We then allocate space for the output in the GPU global-memory using the class method, *copyFromHostToDevice()*. 
```C
    // building output matrix
    const int inputLength = inputArray.inputDimensions[0];
    const int outputLength = inputLength*samplingFactor;
    plhs[0] = mxCreateNumericMatrix(outputLength,
                                    1,
                                    mxDOUBLE_CLASS,
                                    mxREAL);
    CustomGPUObject outputArray(plhs[0]);
    outputArray.copyFromHostToDevice();
```

<!-- ======================================================================= -->
To call the upsampling kernel, we need to setup the launch-configuration parameter. The block-configuration is kept fixed. The number of blocks in the grid is chosen in such a way that the number of blocks are greater than or equal to the number of input-length. We then launch the kernel into the default stream using the launch-configuration parameters and the function arguments.
```C
// calling the upsampling kernel
    dim3 blockConfiguration_upsampling(32,1,1); 
    dim3 gridConfiguration_upsampling((int)((inputLength+blockConfiguration_upsampling.x-1)/blockConfiguration_upsampling.x),\
        1,\
        1);
    upsample_cuda<<<gridConfiguration_upsampling,
                    blockConfiguration_upsampling>>>
                        (inputArray.d_inputPointer_real,
                         samplingFactor,
                         outputArray.d_inputPointer_real,
                         inputArray.d_inputDimensions);
```

<!-- ======================================================================= -->
Next, we update the dimensions of the input-array since the input-array to the next process is the output-array of the previous operation. The outputs of the previous process is then copied into a new vector so that the second process can be overwritten into the same memory.(This might be a wee bit confusing but stay with me)
```C
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
```

<!-- ======================================================================= -->
Next, we call the convolution kernel. First, we setup the launch-configuration parameters. The number of threads per block is the length of the filter-length. The number of blocks is chosen to be the length of the output since each block is logically assigned the responsibility of calculating the value of each sample of the output. Once the launch-configuration parameters are ready, the kernel is launched into the default stream with the launch-configuration parameters and the function arguments. 
```C
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
```

<!-- ======================================================================= -->
Note that kernel-launches towards the default stream is blocking, which means that the lines after the function call are implemented only after the completion of the stream. So, next, we copy the results from the device to the host using the class method, *copyFromDeviceToHost()*.
```C
    // copying data from device to host
    outputArray.copyFromDeviceToHost();
```

### Final CUDA Code
Putting it all together, we get the following


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
    int finalOutputLength = (int)d_inputDimensionsA[0] + \
                            (int)d_inputDimensionsB[0] - 1;
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


<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
## References
- MathWorks. “mxCreateNumericMatrix.” MATLAB API Reference, MathWorks, [Link](www.mathworks.com/help/matlab/apiref/mxcreatenumericmatrix.html). Accessed 26 Sept. 2024.
- “Interpolation.” Wikipedia: The Free Encyclopedia, Wikimedia Foundation, 26 Sept. 2024, [Link](en.wikipedia.org/wiki/Interpolation). Accessed 26 Sept. 2024.







