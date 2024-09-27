# Decimation
Decimation is a Signal Processing technique that reduces the sampling-rate of a digital discrete-time signal by applying a low-pass filter before down-sampling. The primary rate of decimation is to decrease the amount of data while preserving the essential characteristics of the original signal. 

This process involves two stage: filtering and down-sampling. First, the signal is passed through a low-pass filter to limit its bandwidth and eliminate high-frequency components that would've caused aliasing when sampling-rate is reduced. The filtered signal is then downsampled. Mathematically, 

Mathematically, decimation is a two step process 

$$
x_{filtered} = \mathcal{F}^{-1}(H_{lowpass\_filter}(f) \cdot F_{input}(f))\\
x_{decimated }(i, j) = x_{filtered}(i*n, j*n)
$$

where
- $F_{input}(f)$ is the fourier transform of the input signal
- $H_{lowpass\_filter}(f)$ is the fourier transform of the anti-aliasing filter
- $\mathcal{F}^{-1}$ denotes the inverse Fourier transform. 

Decimation is widely used in various applications such as digital communications, audio signal processing, and data compression. In digital communication systems, decimation is employed to reduce the data rate of signals, making them more efficient for transmission and storage. For instance, in a digital radio receiver, the received signal might be decimated to match the sampling rate required by subsequent processing stages, such as demodulation and decoding. In audio signal processing, decimation is used to convert high-sample-rate audio recordings to lower sample rates for efficient storage and playback, while preserving the audio quality. For example, a studio-quality audio signal sampled at 96 kHz may be decimated to 48 kHz for distribution, which reduces the file size and data rate while maintaining sufficient audio fidelity. Decimation is also essential in multi-rate signal processing systems, where signals are processed at different sampling rates for various tasks, such as filtering, analysis, and synthesis.

<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->

## Matlab Code

In the Matlab code, as usual, we proceed to do the following steps
<div style="margin-top: -3mm;"></div>

1. Compile CUDA code
2. Setup the arguments
3. Function call
4. Exhibit results

```Matlab
%% Compiling Code
mexcuda decimate_cuda.cu;
```

<!-- ======================================================================= -->
Next, we prepare the arguments to the function. First, we decide on the sampling-factor. Then, we setup a signal that contains primarily two frequencies:  20Hz and 6000Hz. This is used to demonstrate low-pass filtering. So such wide frequency components makes it easier to see the differences. After the signal creation, we create a low-pass filter, which is necessary before down-sampling. For the low-pass filter, we use a simple FIR, low-pass filter of order,11.

```Matlab
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
```

<!-- ======================================================================= -->
Next, we call the function, receive the output and present the before and after.
```Matlab
%% Calling Function
outputArray = decimate_cuda(inputArray, filterKernel,samplingFactor);

%% Plotting Results
figure(1);
subplot(2,1,1); plot(inputArray); title("Input Signal");
subplot(2,1,2); plot(outputArray); title("Filtered and decimated Signal");
```


<!-- ======================================================================= -->
### Final Matlab Code
Putting it all together, we should get the following



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

<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
## CUDA Code
For the CUDA code, we define two functions: gateway function and decimation kernel. The decimation kernel is a more optimized combination of convolution and decimation. Instead of doing it sequentially, we first identify the samples that will be maintained after down-samplign and just find the convolution result of that particular index. This allows us to avoid finding the convolution result of samples that will be thrown out.

### Decimation Kernel Definition
<!-- ======================================================================= -->
The kernel takes in the following sets of inputs
```C
// kernel
__global__ void conv(double *d_inputPointerA,
                     double *d_inputPointerB,
                     const int samplingFactor,
                     double *d_outputPointer,
                     const mwSize *d_inputDimensionsA,
                     const mwSize *d_inputDimensionsB)
{
    ...
}
```

<!-- ======================================================================= -->
Since our code uses shared-memory, the pointer to the memory needs to be declared. And since the kernel-size cannot be known before hand, we use dynamic shared-memory for this task. 
```C
    // shared memory
    extern __shared__ double sharedMem[];
```

<!-- ======================================================================= -->
Next, we setup some parameters that are required for the computation of the convolution. The are 
1. *inputLength*: the length of the input-signal
2. *indexAssignedToBlock*: The index of input that the current block is responsible for
3. *indicesOfKernelUtilized*: The index up to whos values are considered to find the output for the current output-index. 


```C
    // length of input-array
    const int inputLength = (int)(d_inputDimensionsA[0]); 

    // the index for which we're calculating the result for
    const int indexAssignedToBlock = (int)(blockIdx.x*samplingFactor); 

    // the farthest index from the index, for which this block is calculating dot product for
    int indicesOfKernelUtilized = blockIdx.x - d_inputDimensionsB[0]+1; 
    indicesOfKernelUtilized = min((int)indicesOfKernelUtilized, (int)d_inputDimensionsB[0]);
```

<!-- ======================================================================= -->
Next, we check if the inner-product the current thread is calculating is within the bound of the input-array. Note that this is expected and thus, we need a test, because it is often the case that the number of threadss allocated is greater than the number of elements. After checking validity, we find the index assigned to this thread. The validity of this is also tested. And if it is valid, their element-wise product is calculated and stored into the shared-memory. If it is not valid, we store zero so that the accumulation operation does not work with corrupted values that will end up in wrong results. After accumulation, the first thread of each block is assigned to accumulate the values of the shared-memory of that block. And since each block is assigned the responsibility for each output-index, this results in the value to be stored at that particular output-index. So, the accumulated value is stored into the output-array in the global memory. 
```C
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
```

<!-- ======================================================================= -->
Putting it together, the convolution kernel-definition should look like the following
```C
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
```



### Gateway Function
<!-- ======================================================================= -->
The first stage in the gateway function is check the validity of the inputs. We check the number of inputs, expected number of outputs and the data-type of the inputs. This si done in the following manner
```C
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

    ...
}
```

<!-- ======================================================================= -->
Once the validity of the inputs have been checked, we encapsulate the inputs with objects of the class, *CustomGPUObject*. This data is the made available in the GPU global-memory using the class-method, *copyFromHostToDevice()*. Note that we do not use the class to encapsulate the sampling-factor because it is a scalar. One does not need to explicitly call CUDA-API calls to make scalars available in the GPU global-memory. 
```C
    // Fetching the Input Data
    CustomGPUObject inputArray(prhs[0]);
    CustomGPUObject kernelArray(prhs[1]);
    const int samplingFactor = (int)mxGetScalar(prhs[2]);

    // Transferring data to GPU
    inputArray.copyFromHostToDevice();
    kernelArray.copyFromHostToDevice();
```

<!-- ======================================================================= -->
Next, we setup and calculate a number of parameters required for carrying out this transformation. They are 
1. *inputLength*: length of the input-array
2. *kernelLength*: length of the filter
3. *outputLength*: length of the final output signal

We then use the function, [*mxCreateNumericMatrix*](https://www.mathworks.com/help/matlab/apiref/mxcreatenumericmatrix.html), to create a matrix that will be used to store the output. The created matrix is then encapsulated using an object of the class, *CustomGPUObject*. We then allocate space in the GPU global-memory using the class-method, *copyFromHostToDevice()*. 
```C
    // setup output
    const int inputLength = (int)inputArray.inputDimensions[0];
    const int kernelLength = (int)kernelArray.inputDimensions[0];
    const int outputLength = (int)((inputLength+samplingFactor-1)/samplingFactor);
    plhs[0] = mxCreateNumericMatrix(outputLength, 
                                    1,
                                    mxDOUBLE_CLASS, 
                                    mxREAL);
    CustomGPUObject outputArray(plhs[0]);
    outputArray.copyFromHostToDevice();
```

<!-- ======================================================================= -->
Now that the inputs and space for the output are available at the GPU global-memory, we shall now prepare for the kernel launch. The block-configuration paramter is set such that the number of threads in a block is the kernel-length. The grid-configuration is set such that the number of blocks in a grid is equal to the number of elements in the output. The kernel is then launched in the default stream using the launch-configuration parameters and the function arguments. 
```C
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

```

<!-- ======================================================================= -->
Kernel launches to the default stream is blocking, which means that the lines after this kernel launch is implemented only after the default stream has finished running. Thus, in the next line, we bring the output results from the device memory to the host memory using the class method, *copyFromDeviceToHost()*.
```C
    // Copying results from device to host
    outputArray.copyFromDeviceToHost();
```

### Final CUDA Code
<!-- ======================================================================= -->
Putting it all together, we get the following

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
    outputArray.copyFromHostToDevice();
    
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

<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
## References
- MathWorks. “mxCreateNumericMatrix.” MATLAB API Reference, MathWorks, www.mathworks.com/help/matlab/apiref/mxcreatenumericmatrix.html. Accessed 26 Sept. 2024.
- “Downsampling (Signal Processing).” Wikipedia: The Free Encyclopedia, Wikimedia Foundation, 26 Sept. 2024, en.wikipedia.org/wiki/Downsampling_(signal_processing). Accessed 26 Sept. 2024.













