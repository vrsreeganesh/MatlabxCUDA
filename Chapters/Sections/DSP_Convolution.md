# Standard Convolution

1D convolution is a mathematical operation applied to signals, typically an input signal and a kernel(also called, "filter"), to produce a third signal that represents how the input-signal is modified by the filter. Born from signal processing, it is widely used in neural networks, data analysis and so on. Mathematically, the 1D convolution of an input signal, $x[n]$, with a kernel, $h[k]$, is defined as 

$$
y[n] = (x * h)[n] = \sum_{k=-\infty}^{\infty} x[k] \cdot h[n-k]
$$

Here, $y[n]$ is the resulting output signal, $x[k]$ is the input signal, and $h[n-k]$ is the kernel flipped and shifted by $n$. The sum is taken over all possible values of $k$, indicating that each point in the output signal is a weighted sum of the input signal values, where the weights are given by the kernel.

In practical applications, both the input signal and the kernel are typically finite-length sequences. For a finite-length input signal \( x[n] \) of length \( N \) and a kernel \( h[k] \) of length \( M \), the convolution operation is often simplified and restricted to a finite sum. The equation for the finite-length 1D convolution is:

$$
y[n] = \sum_{k=0}^{M-1} x[n-k] \cdot h[k]
$$

for \( n = 0, 1, 2, \ldots, N + M - 2 \). In this context, appropriate handling of boundary conditions, such as zero-padding the input signal, is necessary to compute the convolution at the edges. This operation produces an output signal \( y[n] \) that is typically longer than the original input signal, with a length of \( N + M - 1 \). 1D convolution is fundamental in many signal processing techniques, including filtering, smoothing, and feature extraction, and serves as a basis for more complex operations in higher dimensions.

<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
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
%% Compiling mex-code
mexcuda conv00.cu
```
<!-- ======================================================================= -->
Next, we prepare the inputs for the function call. To demonstrate convolution, I've chosen to show the process of low-pass filtering. So, the input signal is designed to have two frequency components: high and low. We create the signal in the following manner. 
```Matlab
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
```
<!-- ======================================================================= -->
Next, we setup the filter which will be the second argument. We create a low-pass filter here in the following manner. 
```Matlab
%% Setting up the convolution  kernel
convKernel = transpose(ones(size(1:10)));
convKernel = convKernel/sum(convKernel);
```

<!-- ======================================================================= -->
Now that the inputs have been setup, next we call the function and present the results. 
```Matlab
%% Calling the function
outputArray = conv00(inputArray, convKernel);

%% Plotting before vs after
figure(1);
subplot(1,2,1); plot(inputArray); title("Array: Pre-filtering \n");
subplot(1,2,2); plot(outputArray); title("Array: Post-Filtering \n");
```

<!-- ======================================================================= -->
Putting together all this, we get the following Matlab code.

```matlab
%{
Aim:
    Implementing 1D convolution for audio-signal.
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








<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->

## CUDA 
We need to define two functions to carry out this procedure. The first is the gateway function and the second is the kernel function. 

Convolution operation involves multiple inner-products. The inner-product procedure involves multiple collaborations. Thus, this calls for the use of shared-memory. Ands since the kernel size is never too large, use of shared-memory is highly preferred thanks to low-latency. 

### Kernel Function
The kernel function takes in the following set of arguments
1. The pointer to the input-matrix in the GPU global-memory
2. Pointer to the kernel-matrix in the GPU global-memory
3. Pointer to the memory-allocated for the output in the GPU global-memory. 
4. Pointer to the array storing the dimensions of the input-matrix in the device global-memory. 
5. Pointer to the array storing the dimensions of the kernel-matrix in the device global-memory. 

```C
// kernel
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
In this code, we're opting to use dynamic shared-memory since we don't know the size of the kernel that might be passed as an argument. So in order to enable the use of shared-memory, we shall declare the pointer that will be used to manipulate the elements of the shared-memory in the following manner
```C
    // shared memory
    extern __shared__ double sharedMem[];
```

<!-- ======================================================================= -->
Next, we setup some quantities that will be required for the output. We calculate the following 
<div style="margin-top: -3mm;"></div>

1. The length of the final output
2. The indices of the kernels that will be used to find the output of this particular output-index
3. variable to temporarily store the output
```C
// miscellaneous
    int finalOutputLength = (int)d_inputDimensionsA[0] + (int)d_inputDimensionsB[0] - 1;
    int indicesOfKernelUtilized = blockIdx.x - d_inputDimensionsB[0]+1;
    indicesOfKernelUtilized = min((int)indicesOfKernelUtilized, (int)d_inputDimensionsB[0]);
    double tempOutput;
```

<!-- ======================================================================= -->
Next, we check if the output-index this particular block is assigned to is valid in the following manner
```C
    // checking if we're inside the 
    if(blockIdx.x<finalOutputLength)
    {
        ...
    }
```

<!-- ======================================================================= -->
Next, we calculate the element-wise product of the elements thsi particular thread is responsible for. This involves finding the indices this particular is responsible for by checking the validity. Once the values has been obtained, we find the element-wise product and then store it to the shared-memory in the following manner. 
```C
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
```

<!-- ======================================================================= -->
Once the element-wise products have been calculated and stored, we next assign the first thread to accumulate the values. Note that we call the __syncthreads function before proceeding to ensure that all the threads in the block has completed their element-wise product operations. This ensures that we're accumulating the right values and thereby ensuring the validity of the output. The accumulated value is then stored the memory allocated for the output-array in the following manner.
```C
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
```
<!-- ======================================================================= -->
Putting it together, the final kernel-definition should look like this
```C
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
```

### Gateway Function

<!-- ======================================================================= -->
In the gateway function, we first check the validity of the arugments passed by the Matlab script. This includes testing the number of inputs, number of expected outputs and the data-type of the inputs. This is done in the following manner

```C
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

    ...
}
```
<!-- ======================================================================= -->
Once the validity of the inputs have been verified, we encapsulate the inputs using objects of the class *CustomGPUObject*. This allows for easier manipulation of matrices that are used in both the device and host's memory-space. The obtained results are then sent to the device global-memory in the following manner
```C
    // Fetching the data
    CustomGPUObject inputArray(prhs[0]);
    CustomGPUObject kernelArray(prhs[1]);

    // Sending the Data to the GPU
    inputArray.copyFromHostToDevice();
    kernelArray.copyFromHostToDevice();
```
<!-- ======================================================================= -->
Next, we create a Matlab matrix, using [*mxCreateNumericArray*](www.mathworks.com/help/matlab/apiref/mxcreatenumericarray.html?searchHighlight=mxcreatenumericarray&s_tid=srchtitle_support_results_1_mxcreatenumericarray) that will be used to store the output of this procedure. The matrix, created using this function, is now encapsulated using an object of the class, *CustomGPUObject* and sent to the device global-memory using the class method, *copyFromHostToDevice()*.
```C
    // setup output
    size_t outputLength = inputArray.inputDimensions[0] + kernelArray.inputDimensions[0] - 1;
    plhs[0] = mxCreateNumericMatrix(outputLength,1, mxDOUBLE_CLASS, mxREAL);
    CustomGPUObject outputArray(plhs[0]); 
    outputArray.copyFromHostToDevice();
```
<!-- ======================================================================= -->
Next, we prepare the launch-configuration parameters. The launch-configuration parameters are chosen in such a way that the number of threads within a block is greater than the kernel-size. This is important since only the threads within a block can collaborate using the shared-memory. The number of blocks are chosen based on the length of the output-array. Since we're using shared-memory for this procedure, we need a third-launch configuration parameter, which will be the size of the shared-memory. Thus, we launch the kernel into the default stream using the launch-configuration parameters and the function arguments. 
```C
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
```
<!-- ======================================================================= -->
Kernel launches to the default stream are blocking. Which means that the lines after this call are only implemented after the stream has finished running. Thus, next, we copy the results back from the device memory space to host memory space using the class method, *copyFromDeviceToHost()*. 
```C
    // Fetching Data from GPU to Host
    outputArray.copyFromDeviceToHost();
```
<!-- ======================================================================= -->
### Final CUDA Code
Putting it all together, we get the following CUDA code

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

## References
- MathWorks. “mxCreateNumericArray.” MATLAB API Reference, MathWorks, [Link](www.mathworks.com/help/matlab/apiref/mxcreatenumericarray.html?searchHighlight=mxcreatenumericarray&s_tid=srchtitle_support_results_1_mxcreatenumericarray). Accessed 26 Sept. 2024.
- “Convolution.” Wikipedia: The Free Encyclopedia, Wikimedia Foundation, 26 Sept. 2024, [Link](en.wikipedia.org/wiki/Convolution). Accessed 26 Sept. 2024.











