# Introduction to Shared Memory


## Overview
In this chapter, we introduce shared-memory and how to use it through a very simple example. Shared-memory is a class of on-chip memory in NVIDIA GPUs that is primarily used by threads, within a thread-block, to collaborate. Convolution is a mathematical operation on two functions that produces a third function. The process of convolution calls for the threads to collaborate, due to the accumulative nature of the operation. Thus, making it an excellent example to present the use of shared-memory. 

## Background 

### Convolution Theory
<!--% What is convolution??   -->
Convolution is a mathematical operation on two functions that produces a third function. It is defined as the integral of the product of the two functions after one is reflected about the y-axis and shifted. The integral is evaluated for all values of shift, producing the convolution function.

$$
(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g (t - \tau) d\tau
$$

<!-- why is convolution important?? -->
The process of convolution is important because it serves as a fundamental operation in many fields, including signal processing, image processing, and neural networks. Convolution allows for the systematic combination of two functions or datasets, capturing how the shape of one is modified by the other. In practical terms:

- **Signal Processing**: Convolution helps filter signals by enhancing or suppressing specific features, such as removing noise or highlighting certain frequencies.

- **Image Processing**: In images, convolution is used to apply filters, such as blurring, sharpening, edge detection, and more. This is done by sliding a filter (kernel) over the image to create a new transformed image.

- **Neural Networks**: In convolutional neural networks (CNNs), convolution operations are used to automatically learn and extract features from raw data, like edges, textures, or patterns in images, which are crucial for tasks like image classification or object detection.

In essence, convolution is a versatile tool for transforming data in ways that reveal or emphasize important characteristics, making it invaluable in both analysis and practical applications.



### Shared Memory
Shared Memory is another kind of memory that is available on most NVIDIA-GPUs. Unlike Local Memory and Global memory which reside on-device and far from the SMs, shared memory resides on-chip. Due to the physical proximity in addition to the high-bandwidth channels, the latency for read-and-writes are extremely low and magnitudes lower than that of Global and Local Memory. It is designed to support efficient, high-bandwidth sharing of data among threads in a block. 

Regarding access, the contents of the shared memory are available to all threads within a block. This means that data-structures allocated to shared memory is available to all threads within a block. And each block gets their copy of the structure that was assigned to shared memory. For example, if one were to allocate a double array of length 32, all the blocks will have their own copy of this array. This kind of read-and-write access allows the threads within a block to collaborate to carry out different tasks. 

Shared memory can be assigned in two different ways: static and dynamic. Static allocation allows for higher-dimensional arrays with the trade-off being that the dimensions should be known before-hand. Dynamic allocation, on the other hand, although only allowing for linear arrays, it allows for run-time allocation of shared-memory based on some variable value. To have dynamic high-dimensional arrays, one will have to bring in custom classes that explicitly take care of the expected functionality. The following sections show the two different skeletal ways of allocating shared-memory: static and dynamic. 


#### Shared Memory Allocation: Static
In static allocation, we decide on the size of the available shared memory at compile-time. The standard method of assigning this is through pre-processor directives. Static allocation allows the shared-memory to have higher dimensions. The following shows the creation of a 2-Dimensional shared matrix with the help of preprocessor directives. Note that in static allocation of shared-memory, one does not need to pass the size of the shared-memory as the third kernel-launch parameter. It will be automatically inferred through the preprocessor directives and data-type. 

```C++
// Headerfiles
#include ``mex.h''


// shared-memory related parameters
#define SHAREDMEMX 7
#define SHAREDMEMY 7

// kernel
__global__ mockkernel()
{
	__shared__ sharedMatrix[SHAREDMEMX][SHAREDMEMY];
	...
}

// gate-way function
void mexFunction(...)
{
	...
	// Initialize thread parameters
	dim3 blockspergrid;
	dim3 threadsperblock;
	
	// shared memory related parameters
	int sharedmemoryLength = 128;
	
	// calling the kernel
	kernelfunction<<<blockspergrid, threadsperblock>>>();
	
	
}
```

#### Shared Memory Allocation: Dynamic
In dynamic allocation, we decide on the size of the shared-memory array during run-time. While this allows for flexible decisions regarding size of the object in shared memory, this comes at a trade-off that the array stored in shared memory can only be linear. 

When dynamically allocating shared memory, we just produce the pointer in the kernel. And then we pass the size of the shared memory array through the global-kernel call as the third kernel launch argument. The following shows a skeletal example of how to dynamically allocate a shared array. 

```C++
// Header files
#include "mex.h"

// kernel
__global__ kernelfunction()
{
	__constant__ double sharedMemArray[];
	...
}

// gate-way function
void mexFunction(...)
{
	...
	// Initialize thread parameters
	dim3 blockspergrid;
	dim3 threadsperblock;
	
	// shared memory related parameters
	int sharedmemoryLength = 128;
	
	// calling the kernel
	kernelfunction<<<blockspergrid, threadsperblock, sharedmemorylength*sizeof(double)>>>();
}
```


## MexCUDA Code
The approach we use for this particular problem is to have all the threads in a block produce the output for one particular index. This means that if the output signal is estimated to be of length 128, we allocate 128 total number of blocks. And the number of threads in a block corresponds to the length of the kernel signal, which is usually the signal which is smaller. 

The amount of shared-memory required for this particular operation depends on the arguments. This means that it is ideal to use dynamically allocated shared-memory to perform this operation. And since the inputs are also linear, implementing the operations using linear shared-memory is fairly straightforward. Thus, in the kernel, we start by declaring the pointer to the shared-memory, in the following manner.

```C++
// shared memory
extern __shared__ double sharedMem[];
```

Before we go ahead with the computation, we need to calculate some metrics that we'll need. The first is the final-length of the output. We use the standard equation of convolution to obtain this particular length, which is given by $length(Output) = length(inputA) + length(inputB) - 1$

We then calculate the indices of the kernel that is used. So during convolution, depending on the  index for which we're calculating the values for, we need to decide the indices that are being used. We also declare a variable that stores the value of the product of two values in the array. These steps are done in the following manner. 
```C++
// miscellaneous
int finalOutputLength = (int)d_inputDimensionsA[0] + (int)d_inputDimensionsB[0] - 1;
int indicesOfKernelUtilized = blockIdx.x - d_inputDimensionsB[0]+1;
indicesOfKernelUtilized = min((int)indicesOfKernelUtilized, (int)d_inputDimensionsB[0]);
double tempOutput;
```

Each block is assigned the responsibility of calculating the output for each index. Depending on the position of the index and the number of threads in the block, not all the threads will be utilized for calculating the results of the job. For example, consider the case of calculating the output of index = 0. In this case, only one thread will be utilized since the dot-product is between sub-arrays of length, 1. Thus, we need to add a checkpoint which essentially checks if the thread that is executing it is assigned a responsibility. This is done in the following manner. 

```C++
// checking block-validity
if(blockIdx.x<finalOutputLength)
{
    ...
}
```

Once the check is complete, we assign the threads to calculate the element-wise product of the subarrays required to calculate the outputs of the current index. 

```C++
// checking block-validity
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

    ...
}
```


We now accumulate the element-wise products to obtain the dot-product of the sub-arrays. There is a parallel approach to accumulating values but for now, we assign the accumulation operation to the first thread in the thread-block. The result is then stored into the appropriate index. 

```C++
// checking block-validity
if(blockIdx.x<finalOutputLength)
{
    ...

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

Putting it all together, the kernel function should look like the following. 
```C++
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

    // checking block-validity
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


For the mex gateway-function, we start by first testing the validity of the inputs by testing the number of arguments, data type and dimensionality. 
```C++
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

Once the argument testing has been passed, we receive the inputs into objects of the class, CustomGPUObject. The input arguments are then sent to the device global-memory. 
```C++
// Fetching the data
CustomGPUObject inputArray(prhs[0]);
CustomGPUObject kernelArray(prhs[1]);

// Sending the Data to the GPU
inputArray.copyFromHostToDevice();
kernelArray.copyFromHostToDevice();
```

We then setup space for the output. To do this, we first infer the length of the output of the convolution operation. This is followed by creating a matlab matrix of the same dimensions using the function *mxCreateNumericMatrix*. We use another object of the class, *CustomGPUObject* to encapsulate the output matrix. Even though this data doesn't need to be sent to the device global memory, we use the class method, *copyFromHostToDevice* so that the dimensions are available at the global memory in addition to allocating space in the global memory. 
```C++
// setup output
size_t outputLength = inputArray.inputDimensions[0] + kernelArray.inputDimensions[0] - 1;
plhs[0] = mxCreateNumericMatrix(outputLength,1, mxDOUBLE_CLASS, mxREAL);
CustomGPUObject outputArray(plhs[0]); 
outputArray.copyFromHostToDevice(); 
```

To launch the kernels, we must first prepare the launch-configuration parameters. In our case, since we're using dynamic shared-memory too, in addition to the *gridConfiguration* and *blockConfiguration*, we'll need to provide shared-memory size too. Our kernel is written in such a way that each block is responsible for the output at each index. Thus, this means that the number of threads assigned to a block must be greater than or equal to the number of elements in the kernel. And due to the same reason of each block being assigned the output at each index, the number of blocks will correspond to the length of the output of the convolution operation. 

Once the *gridConfiguration* and *blockConfiguration* has been setup, we perform the kernel launch in the following manner with the third launch-configuration parameter being the size of the shared-memory. 

```C++
// Preparing and calling kernel
dim3 blockConfiguration(kernelArray.inputDimensions[0], 1, 1);
dim3 gridConfiguration(outputLength,1,1);
conv<<<gridConfiguration, 
        blockConfiguration,
        kernelArray.inputDimensions[0]*sizeof(double)>>>(inputArray.d_inputPointer_real,
                                                                kernelArray.d_inputPointer_real,
                                                                outputArray.d_inputPointer_real,
                                                                inputArray.d_inputDimensions,
                                                                kernelArray.d_inputDimensions);
```


The kernel launch in the above case is blocking, which means that the lines after the kernel launch will be executed only after the call has been completed. So we can copy the results back without waiting. Since we're using the class to encapsulate everything, we can fetch the results back from the device global memory to host memory using the class method, *copyFromDeviceToHost* in the following manner. 

```C++
// Fetching Data from GPU to Host
outputArray.copyFromDeviceToHost();
```

% shutting down ----------------------------------------------------------------
This is followed by the usual lines when finishing the use of gpu devices. 
```C++
// Shutting down
cudaDeviceSynchronize();
cudaDeviceReset();
```

## Matlab Code
Now that we've setup the cuda-code for this, we shall start by compiling the code. 
```MATLAB
%% Basic Setup
clc; clear; close all; 

%% Compiling mex-code
mexcuda conv00.cu
```

For demonstrating convolution, we use an averaging kernel with a signal that is made of two frequencies: 50Hz and 7KHz. They are setup in the following manner. 
```MATLAB
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
```


Now that the inputs are setup, we call the mexcuda function and pass the arguments in the function call. The results are then plotted. 
```MATLAB
%% Calling the function
tic
outputArray = conv00(inputArray, convKernel);
toc

%% Plotting before vs after
figure(1);
subplot(1,2,1); plot(inputArray); title("Array: Pre-filtering \n");
subplot(1,2,2); plot(outputArray); title("Array: Post-Filtering \n");
```
