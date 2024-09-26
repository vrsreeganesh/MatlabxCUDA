# Downsampling

Image downsampling is the process of reducing the resolution of an image by decreasing the number of pixels. This is typically achieved by removing samples from the image evenly. Note that this process is different decimation where processing is carried out before downsampling resulting in the image maintaining its visual properties. Downsampling often causes artifacts while decimation does not. Mathematically, the transformation is as follows

$$
I_{down\ sampled}(i, j) = I_{input}(i*n, j*n)
$$

Downsampling is the last step in classical image decimation. So where classical decimation is used, downsampling is also used. 
In digital photography, downsampling is employed to create thumbnail images for quick previews. It is also used in image compression techniques to reduce the amount of data required to represent an image, making storage and transmission more efficient. Another application is in multi-scale image analysis, such as in object detection and recognition tasks, where images at different resolutions are analyzed to capture features at various scales. For example, in medical imaging, downsampling can be used to quickly review large sets of images before examining them in full detail. In each of these cases, downsampling helps manage data more efficiently without significantly compromising the essential information in the image.

<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->

## Matlab Code
Before we prepare the arguments and call the function, the first step is to compile the CUDA code so that it produces the executable, which is what we'll be calling from our code. Note that this does **NOT** need to be compiled before the function is to be called. Ideally, the function just needs to be compiled once its completed or any changes has been made. The compiling-line is thrown into the script purely for simplicity and is often a good practice when developing, as any errors would be immediately brought to your attention.

```Matlab
%% Compiling image
mexcuda downsampling_ip_cuda.cu
```

<!-- ====================================================================== -->
To create the image argument, feel free to read in any image of your liking (3-channels) and feed it as the argument to the function. In our example, however, we use Matlab's *phantom* function to generate an image. The function, essentially, generates a head-phantom that is often used to test the numerical accuracy of image reconstruction algorithms. Though not relevant to our task, this example uses it because my intention is to get you, the reader, to be able to run the code with minimal external dependencies. And the phantom function is available in the base library of Matlab. So, we stick to *phantom*. If this function piques your interest, feel free to read up more at [phantom Matlab Documentation](https://www.mathworks.com/help/images/ref/phantom.html#d126e261153).
```Matlab
%% Preparing Input
inputImage = phantom(512);
inputImage = repmat(inputImage, [1,1,3]);
downsamplingFactor = 4;
```
<!-- ======================================================================= -->
Now, we call the function with the prepared arguments. The returned results are then plotted to compare the input and the output. 

```Matlab
%% Calling Function
outputImage = downsampling_ip_cuda(inputImage, downsamplingFactor);

%% Plotting 
figure(1);
subplot(1,2,1); imagesc(inputImage); title("Input Image");
subplot(1,2,2); imagesc(outputImage); title("Downsampled Image");    
```


### Full Matlab Code
Putting it together, the final code should look like the following



```Matlab
%{
    Aim: Demostrating Downsampling using custom CUDA-kernels
%}

%% Basic setup
clc; clear; close all;

%% Compiling image
mexcuda downsampling_ip_cuda.cu

%% Preparing Input
inputImage = phantom(512);
inputImage = repmat(inputImage, [1,1,3]);
downsamplingFactor = 4;

%% Calling Function
outputImage = downsampling_ip_cuda(inputImage, downsamplingFactor);

%% Plotting 
figure(1);
subplot(1,2,1); imagesc(inputImage); title("Input Image");
subplot(1,2,2); imagesc(outputImage); title("Downsampled Image");    
```










## CUDA Code
For CUDA code, we mainly define two bodies: the kernel and the main function. 

### Kernel 
The kernel takes in the following sets of arguments
1. The pointer to the input-matrix
2. sampling factor
3. Pointer to the memory allocated for output
4. Pointer to the array containing input-dimensions
5. Pointer to the array containing output-dimensions  

So its defined as follows
```C
// kernel
__global__ void downsampling_ip_cuda(double *d_inputPointerA,
                                     const int samplingFactor,
                                     double *d_outputPointer,
                                     const mwSize *d_inputDimensionsA,
                                     mwSize *d_outputDimensions)
{
    ...
}
```

We first create mappings from the thread address to the data-address in the following manner
```C
    // address 
    const int tidx = threadIdx.x + blockIdx.x*blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y*blockDim.y;
    const int tidz = threadIdx.z + blockIdx.z*blockDim.z;
```

Next, we map the input-array indices to the output-array in the following manner. We also test the validity so that wrong accessses or rights are not made. 
```C
    // checking validity
    if(tidx<d_outputDimensions[0] && tidy<d_outputDimensions[1])
    {
        // Indexing into the input
        const int tidx_input = tidx*samplingFactor;
        const int tidy_input = tidy*samplingFactor;
        const int tidz_input = tidz;
        const int linearIndex2D_Input = \
                        tidx_input + \
                        tidy_input*d_inputDimensionsA[0] + \
                        tidz_input*d_inputDimensionsA[0]*d_inputDimensionsA[1];
       
       ...
    }
```

<!-- ======================================================================= -->
We then obtain linear indexes into the output array from the output-index. This is then followed by copying from the input-index to the output-index
```C
        // indexing into the output
        const int linearIndex2D_Output = tidx + \
                                         tidy*d_outputDimensions[0] + \
                                         tidz*d_outputDimensions[0]*d_outputDimensions[1]; 

        // copying the value
        d_outputPointer[linearIndex2D_Output] = (double)d_inputPointerA[linearIndex2D_Input];
```

<!-- ======================================================================= -->
The final kernel definition should look like this

```C
// kernel
__global__ void downsampling_ip_cuda(double *d_inputPointerA,
                                     const int samplingFactor,
                                     double *d_outputPointer,
                                     const mwSize *d_inputDimensionsA,
                                     mwSize *d_outputDimensions)
{
    // address 
    const int tidx = threadIdx.x + blockIdx.x*blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y*blockDim.y;
    const int tidz = threadIdx.z + blockIdx.z*blockDim.z;

    // checking validity
    if(tidx<d_outputDimensions[0] && tidy<d_outputDimensions[1])
    {
        // Indexing into the input
        const int tidx_input = tidx*samplingFactor;
        const int tidy_input = tidy*samplingFactor;
        const int tidz_input = tidz;
        const int linearIndex2D_Input = \
                        tidx_input + \
                        tidy_input*d_inputDimensionsA[0] + \
                        tidz_input*d_inputDimensionsA[0]*d_inputDimensionsA[1];

        // indexing into the output
        const int linearIndex2D_Output = tidx + \
                                         tidy*d_outputDimensions[0] + \
                                         tidz*d_outputDimensions[0]*d_outputDimensions[1]; 
                                         
        // copying the value
        d_outputPointer[linearIndex2D_Output] = (double)d_inputPointerA[linearIndex2D_Input];
    }

    // synchronizing
    __syncthreads();
}

```




### Main
As usual, we start by first checking the validity of the arguments passed. We check the number of inputs, the number of expected outputs and the data-type of the inputs
```C
// gateway Function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // checking number of inputs
    if(nrhs!=2)
        mexErrMsgTxt("Number of Inputs are Wrong \n");
    
    // checking number of outputs
    if(nlhs!=1)
        mexErrMsgTxt("Number of Expected Outputs are Wrong \n");
    
    // checking input data type
    if(!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
        mexErrMsgTxt("The First Argument is of the wrong data-type \n");
    if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]))
        mexErrMsgTxt("The First Argument is of the wrong data-type \n");

     ...
}
```

Once the validity has been checkd, we copy the results in to the appropriate data-structure. Note that we don't use an object of the class, *CustomGPUObject* for the sampling-factor, because its a scalar and scalars can be made available in the GPU memory-space without any CUDA API calls. We then make the input-image available in the GPU global-memory using the method, *copyFromHostToDevice()*. 
```C
    // Fetching Inputs
    CustomGPUObject inputImage(prhs[0]);
    const int samplingFactor = (int)mxGetScalar(prhs[1]);

    // Sending the Data to the GPU
    inputImage.copyFromHostToDevice();
```

Next, we setup the dimensions of the output in the following manner
```C
    // setting up outputs
    mwSize *outputDimensions = (mwSize *)malloc(inputImage.numDimensions*sizeof(mwSize));
    outputDimensions[0] = (mwSize)((inputImage.inputDimensions[0]+samplingFactor-1)/samplingFactor);
    outputDimensions[1] = (mwSize)((inputImage.inputDimensions[1]+samplingFactor-1)/samplingFactor);
    outputDimensions[2] = inputImage.inputDimensions[2];
```

Next, we create a Matlab matrix using the function [*mxCreateNumericArray*](https://www.mathworks.com/help/matlab/apiref/mxcreatenumericarray.html?searchHighlight=mxcreatenumericarray&s_tid=srchtitle_support_results_1_mxcreatenumericarray). This created matrix is then used to create an object of our class, *CustomGPUObject*. We use this object instead of the pointer directly to make coding easier. As usual, since our work is on the GPU, we use the class method, *copyFromHostToDevice* to make memory for the output and sending the output-dimensions to the GPU memory.
```C
    // constructing the output
    plhs[0] = mxCreateNumericArray(inputImage.numDimensions,
                                   outputDimensions, 
                                   mxDOUBLE_CLASS, 
                                   mxREAL);
    CustomGPUObject outputImage(plhs[0]);
    outputImage.copyFromHostToDevice();
```
<!-- ======================================================================= -->
Next, we prepare for the function call in the following manner. The block-size is fixed for general optimality reasons. The number of blocks along each dimensions are chosen in such a way that the number of threads along each dimension is greater or equal to the number of pixels/elements along each dimension. 
```C
    // preparing for function call
    dim3 blockConfiguration(32,32,1);
    dim3 gridConfiguration((int)((outputDimensions[0]+blockConfiguration.x-1)/blockConfiguration.x),
                           (int)((outputDimensions[0]+blockConfiguration.x-1)/blockConfiguration.x),
                            max(1, (int)inputImage.inputDimensions[2]));
```

<!-- ======================================================================= -->
We then launch the kernel with the appropriate arguments and launch-configuration parameters. 
```C
    // calling function
    downsampling_ip_cuda<<<gridConfiguration, 
                           blockConfiguration>>>(inputImage.d_inputPointer_real,
                                                 samplingFactor,
                                                 outputImage.d_inputPointer_real,
                                                 inputImage.d_inputDimensions,
                                                 outputImage.d_inputDimensions);

```
<!-- ======================================================================= -->
Kernel launches to the default stream is blocking. Which means that the rest of the code is run only after the function has finished its processing. So the next line involves copying the results back to make it available in the host-side. This is followed by some maintenance functions. 
```C
    // fetching data from device global-memory to device memory
    outputImage.copyFromDeviceToHost();

    // shutting system down
    cudaDeviceSynchronize();
    cudaDeviceReset();
```

### Final CUDA Code
Putting together the CUDA code, we get the following

```C
/*
    Aim: Implementing down-sampling with CUDA Kernel
*/ 

// header-file
#include "mex.h"
#include "../Beamforming/booktools.h"
#include<cuda_runtime.h>

// kernel
__global__ void downsampling_ip_cuda(double *d_inputPointerA,
                                     const int samplingFactor,
                                     double *d_outputPointer,
                                     const mwSize *d_inputDimensionsA,
                                     mwSize *d_outputDimensions)
{
    // address 
    const int tidx = threadIdx.x + blockIdx.x*blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y*blockDim.y;
    const int tidz = threadIdx.z + blockIdx.z*blockDim.z;

    // checking validity
    if(tidx<d_outputDimensions[0] && tidy<d_outputDimensions[1])
    {
        // Indexing into the input
        const int tidx_input = tidx*samplingFactor;
        const int tidy_input = tidy*samplingFactor;
        const int tidz_input = tidz;
        const int linearIndex2D_Input = \
                        tidx_input + \
                        tidy_input*d_inputDimensionsA[0] + \
                        tidz_input*d_inputDimensionsA[0]*d_inputDimensionsA[1];

        // indexing into the output
        const int linearIndex2D_Output = tidx + \
                                         tidy*d_outputDimensions[0] + \
                                         tidz*d_outputDimensions[0]*d_outputDimensions[1]; 
                                         
        // copying the value
        d_outputPointer[linearIndex2D_Output] = (double)d_inputPointerA[linearIndex2D_Input];
    }

    // synchronizing
    __syncthreads();
}

// gateway Function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // checking number of inputs
    if(nrhs!=2)
        mexErrMsgTxt("Number of Inputs are Wrong \n");
    
    // checking number of outputs
    if(nlhs!=1)
        mexErrMsgTxt("Number of Expected Outputs are Wrong \n");
    
    // checking input data type
    if(!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
        mexErrMsgTxt("The First Argument is of the wrong data-type \n");
    if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]))
        mexErrMsgTxt("The First Argument is of the wrong data-type \n");

    // Fetching Inputs
    CustomGPUObject inputImage(prhs[0]);
    const int samplingFactor = (int)mxGetScalar(prhs[1]);

    // Sending the Data to the GPU
    inputImage.copyFromHostToDevice();

    // setting up outputs
    mwSize *outputDimensions = (mwSize *)malloc(inputImage.numDimensions*sizeof(mwSize));
    outputDimensions[0] = (mwSize)((inputImage.inputDimensions[0]+samplingFactor-1)/samplingFactor);
    outputDimensions[1] = (mwSize)((inputImage.inputDimensions[1]+samplingFactor-1)/samplingFactor);
    outputDimensions[2] = inputImage.inputDimensions[2];

    // constructing the output
    plhs[0] = mxCreateNumericArray(inputImage.numDimensions,
                                   outputDimensions, 
                                   mxDOUBLE_CLASS, 
                                   mxREAL);
    CustomGPUObject outputImage(plhs[0]);
    outputImage.copyFromHostToDevice();

    // preparing for function call
    dim3 blockConfiguration(32,32,1);
    dim3 gridConfiguration((int)((outputDimensions[0]+blockConfiguration.x-1)/blockConfiguration.x),
                           (int)((outputDimensions[0]+blockConfiguration.x-1)/blockConfiguration.x),
                            max(1, (int)inputImage.inputDimensions[2]));
    
    // calling function
    downsampling_ip_cuda<<<gridConfiguration, 
                           blockConfiguration>>>(inputImage.d_inputPointer_real,
                                                 samplingFactor,
                                                 outputImage.d_inputPointer_real,
                                                 inputImage.d_inputDimensions,
                                                 outputImage.d_inputDimensions);

    // fetching data from device global-memory to device memory
    outputImage.copyFromDeviceToHost();

    // shutting system down
    cudaDeviceSynchronize();
    cudaDeviceReset();
}
```

## References
- MathWorks. “Phantom.” MATLAB Documentation, MathWorks, [Link](www.mathworks.com/help/images/ref/phantom.html#d126e261153). Accessed 26 Sept. 2024.
- MathWorks. “mxCreateNumericArray.” MATLAB API Reference, MathWorks, [Link](www.mathworks.com/help/matlab/apiref/mxcreatenumericarray.html?searchHighlight=mxcreatenumericarray&s_tid=srchtitle_support_results_1_mxcreatenumericarray). Accessed 26 Sept. 2024.
- “Downsampling (Signal Processing).” Wikipedia: The Free Encyclopedia, Wikimedia Foundation, 26 Sept. 2024, en.wikipedia.org/wiki/Downsampling_(signal_processing). Accessed 26 Sept. 2024.