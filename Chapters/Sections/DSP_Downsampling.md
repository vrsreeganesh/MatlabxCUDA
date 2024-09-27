# Signal Downsampling

Downsampling is a Signal Processing technique used as part of the decimation procedure. Mathematically, if $x[n]$ represents the original signal, the downsampled signal $y[m]$ can be obtained by taking every $M$-th sample of $x[n]$:

$$
y[m] = x[mM]
$$

where $M$ is the downsampling factor. This operation reduces the length of the signal by a factor of $M$, resulting in a new signal that is $1/M$ the length of the original signal. 

Note that downsampling and decimation are very important. Decimation involves first processing the signal before downsampling. Just downsampling a signal results in aliasing, producing artifacts and thereby significant loss of audio-quality. 

Since downsampling is an integral part of classical signal-decimation, it is used wherever classical signal-decimation is employed. For example, in audio processing, downsampling is used to decrease the file size of audio recordings, making them more suitable for storage or streaming. For instance, an audio file sampled at 44.1 kHz can be downsampled to 22.05 kHz, which reduces the file size by half while still retaining sufficient quality for many applications. 


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
%% Compiling input code
mexcuda downsample_cuda.cu
```

<!-- ======================================================================= -->
Next, we prepare the inputs for the function. We create a simple sinusoidal signal for the first input. For the sampling factor, we choose, 10. 
```Matlab
%% preparing inputs
inputArray = sin(2*pi*1e2*linspace(0,1,2048));
inputArray = inputArray(:);
samplingFactor = 10;
```

<!-- ======================================================================= -->
Next, we call the function, receive the output and present the before and after in the following manner. 
```Matlab
%% Calling the functions
outputArray = downsample_cuda(inputArray, samplingFactor);

%% displaying the results
figure(1);
subplot(2,1,1); plot(inputArray); title("Input Array");
subplot(2,1,2); plot(outputArray); title("Output Array");
```

<!-- ======================================================================= -->
### Final Matlab Code
Putting it all together, we get the following

```matlab
%% Aim:Demonstrating CUDA implementation of down-sampling. 

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

<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
## CUDA Code
For the CUDA code, we define two functions: the gateway function and the kernel-definition. 


### Kernel Definition
<!-- ======================================================================= -->
The kernel takes in the following set of inputs
<div style="margin-top: -3mm;"></div>

1. Pointer to the input-array in the GPU global-memory
2. sampling-factor
3. Pointer to the memory-allocated for output, in the GPU global-memory. 
4. Pointer to the array holding the input-dimensions

```C
// global kernel
__global__ void downsample_cuda(double *d_inputPointerA,
                                const int samplingFactor,
                                double *d_outputPointer,
                                const mwSize *d_inputDimensionsA)
{
    ...
}
```

<!-- ======================================================================= -->
We first produce the mappings from the thread-address to the indices of the output data-structure
```C
    // getting global address
    const int tidx = threadIdx.x + blockIdx.x*blockDim.x;
```

<!-- ======================================================================= -->
From the output-indx, we calculate the input index in the following manner. We also obtain the length of the input, which will be later used to verify the validity of indices. 
```C
    // miscellaneous
    const int responsibleIndex = tidx*samplingFactor;
    const int inputLength = (int)d_inputDimensionsA[0];
```


<!-- ======================================================================= -->
Next, we carry out the sampling procedure according to the equation presented before
```C
    /*
    If the index from which we're fetching values are within
    the length of the array, then we copy it
    */ 
    if(responsibleIndex<inputLength)
    {
        // copying the values
        d_outputPointer[tidx] = (double)d_inputPointerA[responsibleIndex];
    }
```

<!-- ======================================================================= -->
Putting it all together, the kernel definition should look like the following
```C
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
```


### Gateway Function

<!-- ======================================================================= -->
In the gateway function, we first verify the validity of the inputs. The check includes the number of inputs, the number of expected outputs and the data-type of the inputs.
```C
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

    ...
}
```


<!-- ======================================================================= -->
Once the validity of the inputs have been confirmed, we encapsulate the inputs using an object of the class, *CustomGPUObject*. The data is then made available in the device global-memory in the following manner. Note that we do not use an object to encapsulate the scalar because making scalars available in the GPU memory-space requires no special explicit CUDA-API calls. 
```C
    // Fetching Inputs 
    CustomGPUObject inputArray(prhs[0]);
    const int samplingFactor = (int)mxGetScalar(prhs[1]);

    // Transferring Assets to the Device
    inputArray.copyFromHostToDevice();
```

<!-- ======================================================================= -->
Next, we setup Matlab matrices to store the output of the procedure. This involves first calculating the dimensions of the outputs and then creating the matrix using the function, [*mxCreateNumericArray*](https://www.mathworks.com/help/matlab/apiref/mxcreatenumericarray.html). The created matrix is then encapulated using an object of the class, *CustomGPUObject* and then made available in the GPU memory-space using the class method, *copyFromHostToDevice()*. 
```C
    // setting up output
    const int inputLength = (int)(inputArray.numElements); 
    const int outputLength = (int)((inputLength+samplingFactor-1)/samplingFactor);
    plhs[0] = mxCreateNumericMatrix(outputLength,
                                    1, 
                                    mxDOUBLE_CLASS, 
                                    mxREAL);
    CustomGPUObject outputArray(plhs[0]);
    outputArray.copyFromHostToDevice();
```

<!-- ======================================================================= -->
Once the inputs have been made available in the device global-memory and memory has been allocated for the output in the device global-memory, we prepare for function call. First, we setup the launch-configuration parameters. The block-configuration is kept fixed at the optimal value of 32. The number of blocks are decided in such a way that the number of threads in total is greater than or equal to the number of elements in the output. We launch the kernels into the default stream with the launch-configuration parameters and the function arguments in the following manner. 
```C
    // preparing for function call 
    dim3 blockConfiguration(32,1,1);
    const int numblocks = (int)((outputLength+blockConfiguration.x-1)/blockConfiguration.x);
    dim3 gridConfiguration(numblocks,1,1);
    downsample_cuda<<<gridConfiguration, 
                        blockConfiguration>>>(inputArray.d_inputPointer_real,
                                            samplingFactor,
                                            outputArray.d_inputPointer_real,
                                            inputArray.d_inputDimensions);
```

<!-- ======================================================================= -->
Calls to the default stream is blocking, which means that the lines after this call are executed only after the stream has been completed. Thus, next, we copy the results back from the device to the host using the class method, *copyFromDeviceToHost()*.
```C
    // Copying data from Device to Host
    outputArray.copyFromDeviceToHost();
```

### Final CUDA Code
<!-- ======================================================================= -->
Putting it all together, we get the following result


```C
// Aim: Building custom-kernel to down-sample code. 

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
- "Downsampling (Signal Processing)." Wikipedia, The Free Encyclopedia, 12 Sept. 2024, [Link](en.wikipedia.org/wiki/Downsampling_(signal_processing)). Accessed 12 Sept. 2024.
- MathWorks. “mxCreateNumericArray.” MATLAB API Reference, MathWorks, [Link](www.mathworks.com/help/matlab/apiref/mxcreatenumericarray.html). Accessed 26 Sept. 2024.







