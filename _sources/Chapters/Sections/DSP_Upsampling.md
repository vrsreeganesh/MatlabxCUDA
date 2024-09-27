# Signal Upsampling

Upsampling is a signal-processing technique used to increase the length of a digital signal by inserting additional samples between original ones. This process effectively increases the length and is the first step in signal interpolation, a proocess used to increase the data-rate/sampling-frequency of signals. The main difference between upsampling and interpolation is that interpolation has an additional step of filtering after upsampling. Proper filtering, such as using a low-pass filter, is usually required after upsampling to remove the spectral replicas introduced by the zero-insertion process and to ensure that the upsampled signal accurately represents the original signal.


Mathematically, if $x[n]$ represents the original signal, the upsampled signal $y[m]$ can be obtained by inserting $ L-1$ zeros between each sample of $x[n]$:

$$
y[m] = 
\begin{cases} 
x[m/L] & \text{if } m = kL \text{ for some integer } k \\
0 & \text{otherwise}
\end{cases}
$$

where $L$ is the upsampling factor. This operation increases the sampling rate by a factor of $L$, resulting in a new signal that has $L$ times the length of the original signal. For instance, if $x[n]$ is sampled at 10 kHz and $L = 2$, the upsampled signal $y[m]$ will have a sampling rate of 20 kHz.

Since upsampling is part of every classical interpolation technique, it is used wherever interpolation is used.For example, in digital communication, interpolation is used to match the sampling rate of a signal to the requirements of the transmission medium or the receiver. For example, a signal sampled at a lower rate might be upsampled to a higher rate to meet the specifications of a digital-to-analog converter (DAC). In audio processing, interpolation is used to improve the quality of audio playback by increasing the sampling rate, allowing for a smoother and more accurate representation of the sound. 



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
mexcuda upsample_cuda.cu;
```

<!-- ======================================================================= -->
Next, we prepare the inputs to this function. The first input is the signal that needs to be upsampled and the second argument is the sampling-factor. 
```Matlab
%% preparing inputs
inputArray = 1:100;
inputArray = inputArray(:);
samplingFactor = 7;
```


<!-- ======================================================================= -->
Next, we call the function with the prepared arguments and present the inputs and outputs to see the difference
```Matlab
%% Calling the functions
outputArray = upsample_cuda(inputArray, samplingFactor);

%% displaying the results
figure(1);
subplot(1,2,1); plot(inputArray); title("Input Array");
subplot(1,2,2); plot(outputArray); title("Output Array");
```


<!-- ======================================================================= -->
### Final Matlab Code
Putting it all together, we get the following

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
The kernel takes in the following set of arguments
1. Pointer to the input-array in the GPU global-memory
2. Sampling factor
3. Pointer to the memory allocated for the output array, in the GPU global-memory.
4. Pointer to the array containing the dimensions of the input-array
<!-- ======================================================================= -->

```C
// global kernel
__global__ void upsample_cuda(double *d_inputPointerA,
                                const int samplingFactor,
                                double *d_outputPointer,
                                const mwSize *d_inputDimensionsA)
{
    ...
}
```


<!-- ======================================================================= -->
Next, we assign the threads to the dataa-point it is responsibel for. This is done by producing mappings from the thread addresses to the indices in the following manner.
```C
   // getting global address
    const int tidx = threadIdx.x + blockIdx.x*blockDim.x;
```


<!-- ======================================================================= -->
Next, we setup some important variables. The first is the index that this particular thread is responsible for. The second is a variable that holds the length of the input. This will be used to verify the validity of the memory read and written by the threads. Note that this is necessary because it is very often the case that the number of threads allocated are more than the number of threads required. 
```C
    // miscellaneous
    const int responsibleIndex = tidx*samplingFactor;
    const int inputLength = (int)d_inputDimensionsA[0];
```


<!-- ======================================================================= -->
Next, we perform the processing by copying the appropriate inputs at the input-index to the appropriate output-index.
```C
    // checking if tidx maps to a valid index of the input array
    if(tidx<inputLength)
    {
        // copying the values
        d_outputPointer[responsibleIndex] = (double)d_inputPointerA[tidx];
    }
```


<!-- ======================================================================= -->
Finally, the kernel-definition should look like the following
```C
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

```

### Gateway Function
<!-- ======================================================================= -->
In the gateway function, we first ensure the validity of the input arguments. This involves checking the number of inputs, number of expected outputs and the data-type of the inputs. 
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
Next, we encapsulate the received input with the objects of the class, *CustomGPUObject*. This data is then made available in the GPU device-memory using the class method, *copyFromHostToDevice()*. Note that we do not employ this class for storing the sampling-factor because we don't need explicit CUDA-API calls for transferring scalars from host to device. 
```C
    // Fetching Inputs
    CustomGPUObject inputArray(prhs[0]);
    const int samplingFactor = (int)mxGetScalar(prhs[1]);

    // Sending Data to GPU
    inputArray.copyFromHostToDevice();
```

<!-- ======================================================================= -->
Next, we setup the output. We also initialize some variables that are used to store the input-length and output-length. We create a Matlab matrix using the function, [*mxCreateNumericMatrix*](https://www.mathworks.com/help/matlab/apiref/mxcreatenumericmatrix.html). The created matrix is then encapsulated using an object of the class, *CustomGPUObject*. The data-structure is then made available in the GPU global-memory using the class method, *copyFromHostToDevice()*. 
```C
// setting up output
    const int inputLength = (int)(inputArray.numElements); 
    const int outputLength = (int)(inputLength*samplingFactor);
    plhs[0] = mxCreateNumericMatrix(outputLength,
                                    1, 
                                    mxDOUBLE_CLASS, 
                                    mxREAL);
    CustomGPUObject outputArray(plhs[0]);
    outputArray.copyFromHostToDevice();
```

<!-- ======================================================================= -->
Next, we setup the launch-configuration parameters: block-configuration and grid-configuration. The block-configuration is fixed to be 32. The number of blocks are chosen in such a way that the number of threads are greater than or equal to the number of elements in the input. Once the launch-configuration parameters have been setup, we launch the kernels into the default stream with the launch-configuration parameters and function arguments in the following manner.
```C
    // function call
    dim3 blockConfiguration(32,1,1);
    const int numblocks = (int)((inputLength+blockConfiguration.x-1)/blockConfiguration.x);
    dim3 gridConfiguration(numblocks,1,1);
    upsample_cuda<<<gridConfiguration,
                    blockConfiguration>>>(inputArray.d_inputPointer_real,
                                            samplingFactor,
                                            outputArray.d_inputPointer_real,
                                            inputArray.d_inputDimensions);
```

<!-- ======================================================================= -->
Note that the kernel launches to the default stream is blocking. This means that the lines after this function call will only bet run after the stream has finished running. Thus, after, we copy the results from the device to the host using a class method, *copyFromDeviceToHost()*. 
```C
    // Copying results from device to host
    outputArray.copyFromDeviceToHost();
```

<!-- ======================================================================= -->
### Final CUDA Code
Putting it all together, we get the following
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


<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->

## References
- "Upsampling." Wikipedia, The Free Encyclopedia, 11 Sept. 2024, en.wikipedia.org/wiki/Upsampling. Accessed 12 Sept. 2024.
- MathWorks. “mxCreateNumericMatrix.” MATLAB API Reference, MathWorks, [Link](www.mathworks.com/help/matlab/apiref/mxcreatenumericmatrix.html). Accessed 26 Sept. 2024.
