# Introduction to Constant Memory

## Overview
In this chapter, we introduce constant-memory and how to use it through a very simple example. *Constant Memory* is another class of memory that is available in most NVIDIA GPUs. Using constant-memory to store data that is highly fetched, won't be edited and not-too-large for cache results in highly improved performance due to the reduced data-fetch times from the SMs. This chapter presents how to use constant-memory generally and demonstrate how to use constant-memory through an example. 

## Background

### Stencil Operation 
A stencil operation is a computational technique used in multiple fields like image and signal processing. It involves applying a function or operation to a neighborhood of data points around a central point in a grid or an array. The output of this operation produces values at each point which is a function of  the elements in its neighborhood. 

The following are the key-aspects of a stencil operation 
- **Neighborhood**: The stencil defines a pattern or shape around each point in the grid that forms the neighborhood of that point for this operation. The elements of this neighborhood are used to compute the results that will be stored in the output index corresponding to the input center index. Common stencils for images include 3x3, 5x5 or more complex shapes. Note that the dimensions are usually odd-dimensions along each arrays so that there is a "center element". 
- **Computation**: At each grid point, the operation combines the values of the neighboring points according to the stencil's rules. Basic stencil operations are averaging, summing, max-ing, or more complex functions. 
- **Applications**: Following are three different fields where this is used
    - **Numerical Methods**: In solving partial differential equations (PDEs), stencil operations are used to approximate derivatives and other operations.
    - **Image Processing**: Stencils are applied for tasks like edge detection, blurring, and sharpening, where each pixel's new value depends on the values of its neighboring pixels.
    - **Scientific Computing**: Stencil operations are common in simulations, such as fluid dynamics, where they model the interaction between points in a grid.
Stencil operations are highly parallelizable, making them well-suited for implementation on GPUs using CUDA.

### Constant Memory 
*Constant Memory* is another class of memory that is available in most NVIDIA GPUs. The constant memory is at the same distance from the SM as that of the Global Memory. Though constant memory is not at the same proximity as that of shared-memory, it is aggressively cached. This means that using constant memory to store data that will remain constant throughout the process in addition to being regularly used will produce gains in speed. Thus this kind of data that is stored in the constant memory is efficiently and effectively cached. This kind of aggressive caching allows for extreme streamlining and large gains in time due to the high hit rates in the cache. 

Regarding access-privilege, the data in Constant Memory is only allowed to be set from the host-side and never from the device-side. That is, host side write access while device-side has read access. Due to these technical designs and characteristics, Constant Memory must be used to store those values that are 
- Regularly Used
- No edits
- Not too large

Thus, for tasks such as convolution where we use the same kernel throughout the data, using constant-memory to store the kernel-weights is a good idea. Even though most filters will fit just fine into constant-memory of constant sizes, it is important to remember that the improvement in performance stems from the high-hit rate of increasing the probability of finding them in the cache associated with the constant-memory. 

#### Using Constant-Memory (complete but  not polished)
Unlike Shared Memory, *Constant Memory* doesn't give us the option to be dynamically allocated. Thus, we allocate constant memory during compile time, through preprocessor directives. Constant memory is declared in the same way we declare a global variable, but with the addition that we use the identifier, $\_\_constant\_\_$. Once declared, we populate constant-memory using the function, *cudaMemcpyToSymbol()*. Following is a skeletal code that demonstrates using constant memory. 

```C++
...
// How to Declare Constant Memory 
#define CONSTANT_MEMORY_LENGTH 21
__constant__ double constantFilterKernel[CONSTANT_MEMORY_LENGTH];

// gate-way function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	...
	
	// sending data from Host to Constant Memory
    cudaMemcpyToSymbol(constantFilterKernel,
                       hostFilterKernel,
                       numElementsKernel*sizeof(double));
	...
}

...
```

## MexCUDA Code(complete but not polished)  
We start by declaring constant memory. The following lines of code shows the use of preprocessor directives to choose the size of the constant-memory. 
```C++
// setting up constant memory
#define CONSTANT_MEMORY_LENGTH 21
__constant__ double constantFilterKernel[CONSTANT_MEMORY_LENGTH];
```

### Kernel (complete but not polished)
In the kernel, we start by obtaining the mapping from the thread-address to the data-index that the particular thread is responsible for. For this particular example, we're mapping the thread-address to the data-index that the thread will be responsible for. The mapping to the thread-address to the output-index is done in the following manner. 
```C++
// mapping from thread-address to output-address
size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
```

Note that the stencil operation is different from that of convolution. In the sense that the stencil operation is done in such a way that the stencil is always inside the input array. This means that the output index will always be less than the length of the input array since the output will be of the same length. Thus, the gating is as follows
```C++
// ensuring validity
if(tid<inputDimensionsA[1])
{
    ...
}
```

Before we perform the operation, we setup the variables that will be used for the operation in the following manner. 
```C++
    // ensuring validity
    if(tid<inputDimensionsA[1])
    {
        // performing the kernel operation
        double accusum = (double)0.0;
        int bigIndex = 0;
        int kernelIndex = 0;

        ...
    }
```

Unlike convolution, the output is stored to the position of the middle-index in the output array. So keeping this in mind, we calculate the dot product of the two sub-arrays in the following manner. 
```C++
    // ensuring validity
    if(tid<inputDimensionsA[1])
    {
        ...

        for(int i = -1*(int)(inputDimensionsB[1]/2); i<=(int)(inputDimensionsB[1]/2); ++i)
        {
            // obtaining the absolute indices into each datastructures
            bigIndex = tid + i; 
            kernelIndex = i + (int)(inputDimensionsB[1]/2);

            // checking for boundary conditions
            if(bigIndex>=0 || bigIndex< inputDimensionsA[0])
            {
                accusum = accusum + (double)dA[bigIndex]*(double)constantFilterKernel[kernelIndex];
            }
        }

        ...
    }
```

Once the results have been computed, we store it to the output in the following manner. 
```C++
    // ensuring validity
    if(tid<inputDimensionsA[1])
    {
        ...

        // adding the value to the final result
        dC[tid] = (double)accusum;
    }
```

Putting it together, we get the following 
```C++
// global kernel
__global__ void Stencil_ConstantMemory(double *dA,
                                            double *dC, 
                                            mwSize *inputDimensionsA, 
                                            mwSize *inputDimensionsB)
{
    // addresses
    size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    // ensuring validity
    if(tid<inputDimensionsA[1])
    {
        // performing the kernel operation
        double accusum = (double)0.0;
        int bigIndex = 0;
        int kernelIndex = 0;

        for(int i = -1*(int)(inputDimensionsB[1]/2); i<=(int)(inputDimensionsB[1]/2); ++i)
        {
            // obtaining the absolute indices into each datastructures
            bigIndex = tid + i; 
            kernelIndex = i + (int)(inputDimensionsB[1]/2);

            // checking for boundary conditions
            if(bigIndex>=0 || bigIndex< inputDimensionsA[0])
            {
                accusum = accusum + (double)dA[bigIndex]*(double)constantFilterKernel[kernelIndex];
            }
        }

        // adding the value to the final result
        dC[tid] = (double)accusum;
    }
}
```

### Gateway function
In the gateway function, we first check the number of inputs provided, the number of outputs expected, the data-type and the dimensionality in the following manner. 
```C++
// gateway function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // checking number of inputs
    if(nrhs!=2)
    {
        mexErrMsgTxt("Number of Inputs are Wrong \n");
    }
    
    // checking number of outputs
    if(nlhs!=1)
    {
        mexErrMsgTxt("Number of Outputs are Wrong \n");
    }
    
    // checking input data types
    if(!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) )
    {
        mexErrMsgTxt("One of the Inputs are of the wrong data-type \n");
    }

    ...
}    
```


Once the checking is complete, we fetch the inputs, store it into the objects of the class, *CustomGPUObject* and send it to the device global memory in the following manner. 
```C++
// gateway function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ...

    // Fetching inputs and sending to device global memory
    CustomGPUObject inputA(prhs[0]);
    CustomGPUObject inputB(prhs[1]);

    // Sending inputs to device global memory
    inputA.copyFromHostToDevice();
    inputB.copyFromHostToDevice();

    ...
}
```

Next, we populate the contents of the shared-memory with the stencil weights. This step is done in the following manner.
```C++
// sending input filter to constant memory
cudaMemcpyToSymbol(constantFilterKernel,
                    inputB.inputPointer_real,
                    inputB.numElements*sizeof(double));
```

Next, we create a matlab matrix to store the output. This particular object is then encapsulated using an object of the class, *CustomGPUObject*. The contents of this class is then made available in the global device memory using the class method, *copyFromHostToDevice*. This is done in the following manner. 
```C++
// gateway function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ...

    // setting  up output
    plhs[0] = mxCreateNumericMatrix(1, 
                                    inputA.numElements, 
                                    mxDOUBLE_CLASS, 
                                    mxREAL);
    CustomGPUObject outputC(plhs[0]);
    outputC.copyFromHostToDevice();
    
    ...
}   
```

Once the constant memory has been populated and the rest of the inputs are available in the global memory, we perform the kernel launch in the following manner. First, we seup the launch configuration parameters: block-configuration and grid-configuration. That is followed by kernel launch in the following manner. 
```C++
// gateway function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ...
    
    // prepping to call function 
    dim3 block(1, 1, 1);
    dim3 grid(inputA.numElements,1,1);
    
    // calling function
    Stencil_ConstantMemory<<<grid,
                             block>>>(inputA.d_inputPointer_real, 
                                      outputC.d_inputPointer_real, 
                                      inputA.d_inputDimensions, 
                                      inputB.d_inputDimensions);
    
    ...
}
```

Once the outputs are computed by the kernel and made available in the device global memory, we copy the results back using the class method, *copyFromDeviceToHost*. This is followed by the usual commands to shut down the GPU. 
```C++
// gateway function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ...

    // retrieving data from device to host
    outputC.copyFromDeviceToHost();

    // shutting down
    cudaDeviceSynchronize();
    cudaDeviceReset();
}
```


## Matlab Code (complete but not polished)
In the matlab section, we first start with compiling the cuda code in the following manner. 
```MATLAB
%% Basic Setup
clc; clear; close all;

%% Compiling mex-cuda
mexcuda StencilConstantMemory_CUDA.cu
```

We then prepare simple inputs for this particular example in the following manner.
```Matlab
%% Preparing arguments
inputArray = ones(1,20);
kernelArray = ones(1,11);
```

Once the cuda code has been compiled (doesn't need to be done each time) and the inputs have been prepared, we call the function and get the results in the following manner. We then plot the results. 
```Matlab
%% Calling function
outputArray = StencilConstantMemory_CUDA(inputArray, kernelArray);

%% Plotting 
figure(1);
subplot(2,1,1); plot(inputArray); title("inputArray");
subplot(2,1,2); plot(outputArray); title("outputArray");    
```


