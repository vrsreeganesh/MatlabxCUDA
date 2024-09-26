# Decimation

Image decimation is a technique in Digital Image Processing where the size of a digital image is reduced while maintaing as much information as possible. This is different from downsampling where the samples in between are simply tossed out. In decimation, the image is processed before downsampling. This additional step is called anti-aliasing. Without this step, high-frequency information in the image could cause aliasing, where different signals become indistinguishable, leading to artifacts in the decimated image. The anti-aliasing filter is typically a low-pass filter that smooths the image by reducing high-frequency components. 

Mathematically, decimation is a two step process 

$$
I_{filtered} = \mathcal{F}^{-1}(H_{lowpass\_filter}(f) \cdot F_{input}(f))\\
I_{decimated }(i, j) = I_{filtered}(i*n, j*n)
$$

where 
- $F_{input}(f)$ is the fourier transform of the input image
- $H_{lowpass\_filter}(f)$ is the fourier transform of the anti-aliasing filter
- $\mathcal{F}^{-1}$ denotes the inverse Fourier transform. 


Decimation is used in various contexts such as video processing, where reducing the resolution of frames lowers the bandwidth needed for streaming, and in remote sensing, where it manages the vast data collected by satellite sensors for quicker analysis. In machine learning, particularly in image preprocessing for training models, decimation reduces the input image size, thereby speeding up the training process by lowering computational overhead while preserving essential features for recognition tasks.



<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
## MATLAB Code
In the Matlab code, as usual, we proceed to do the following steps
1. Compile CUDA code
2. Setup the arguments
3. Function call
4. Exhibit results

<!-- ======================================================================= -->
Before we prepare the arguments and call the function, the first step is to compile the CUDA code so that it produces the executable, which is what we'll be calling from our code. Note that this does **NOT** need to be compiled before the function is to be called. Ideally, the function just needs to be compiled once its completed or any changes has been made. The compiling-line is thrown into the script purely for simplicity and is often a good practice when developing, as any errors would be immediately brought to your attention.

```Matlab
%% Compiling Code
mexcuda decimation_ip_cuda.cu
```
<!-- ======================================================================= -->
To create the image argument, feel free to read in any image of your liking (3-channels) and feed it as the argument to the function. In our example, however, we use Matlab's *phantom* function to generate an image. The function, essentially, generates a head-phantom that is often used to test the numerical accuracy of image reconstruction algorithms. Though not relevant to our task, this example uses it because my intention is to get you, the reader, to be able to run the code with minimal external dependencies. And the phantom function is available in the base library of Matlab. So, we stick to *phantom*. If this function piques your interest, feel free to read up more at [phantom Matlab Documentation](https://www.mathworks.com/help/images/ref/phantom.html#d126e261153). We also setup the sampling factor. 
```Matlab
%% Setting up Inputs
inputImage = phantom(2048);
samplingFactor = 4;
```


<!-- ======================================================================= -->
Next, we design the anti-aliasing filter. The anti-aliasing filter we're designing here has a transition-bandwidth of 0.1, filter order = 31. We also design the code such that the end of the passband and the beginning of the stop band is symmetrical to the mid frequency. With those specifications, we create the filter with the function, [*firpm*](https://www.mathworks.com/help/signal/ref/firpm.html) which returns the parameters for [Parks-McClellan optimal FIR filter](https://en.wikipedia.org/wiki/Parks–McClellan_filter_design_algorithm). The function returns the filter for a 1D signal. We do the following operations so that it can be applied to 2D signals, that is images. 
```Matlab
%% Designing an anti-aliasing filter
transitionbandwidth = 0.1;                           
M = 2;                                               
filterorder = 31;                          
endofpassband = (1/M) - transitionbandwidth/2;      % end of passband
startofstopband = (1/M) + transitionbandwidth/2;    % start of stop-band
filtercoefficients = firpm(filterorder-1, ...       % designing our filters
                            [0,endofpassband, startofstopband, 1], ...
                            [1,1,0,0]);
inputKernel = repmat(filtercoefficients, [length(filtercoefficients), 1]) .* repmat(transpose(filtercoefficients), [1, length(filtercoefficients)]);

```

<!-- ======================================================================= -->
Next, we call the CUDA function with the appropriate arguments and present the results side by side. 
```Matlab
%% Running Kernel
outputImage = decimation_ip_cuda(inputImage, inputKernel, samplingFactor);

%% Plotting the Results
figure(1);
subplot(1,2,1); imagesc(inputImage); colorbar; title("Input Image");
subplot(1,2,2); imagesc(outputImage); colorbar; title("Decimated Image");
```

### Final Matlab Code
<!-- ======================================================================= -->
The final Matlab code should look like the following
```Matlab
%{
    Aim: Demonstrating Decimation with Custom CUDA Kernels
%}

%% Basic Setup
clc; clear;

%% Compiling Code
mexcuda decimation_ip_cuda.cu

%% Setting up Inputs
inputImage = phantom(2048);
samplingFactor = 4;

%% Designing an anti-aliasing filter
transitionbandwidth = 0.1;                           
M = 2;                                               
filterorder = 31;                          
endofpassband = (1/M) - transitionbandwidth/2;      % end of passband
startofstopband = (1/M) + transitionbandwidth/2;    % start of stop-band
filtercoefficients = firpm(filterorder-1, ...       % designing our filters
                            [0,endofpassband, startofstopband, 1], ...
                            [1,1,0,0]);
inputKernel = repmat(filtercoefficients, [length(filtercoefficients), 1]) .* repmat(transpose(filtercoefficients), [1, length(filtercoefficients)]);


%% Running Kernel
outputImage = decimation_ip_cuda(inputImage, inputKernel, samplingFactor);

%% Plotting the Results
figure(1);
subplot(1,2,1); imagesc(inputImage); colorbar; title("Input Image");
subplot(1,2,2); imagesc(outputImage); colorbar; title("Decimated Image");
```


<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->

## CUDA
Due to the collaborative nature of Filtering, we'll be using shared-memory for this task. The use of shared-memory means that this must be considered when deciding on the block-configuration, grid-configuration, mapping from thread-address to the data-structure address. 

For the block-configuration, we must ensure that the number of threads in a block is greater than or equal to the number of elements in the kernel. This is because the shared-memory can only be used by the threads in a block. So if the output at a pixel requires the inner-product of the neighbours, the element-wise products of the neighbours and the kernel elements are done in the shared-memory. 

Thus, we're assuming each block to each output-pixel since the threads under the block are used to compute the amplitude of that particular pixel. Thus, the number of blocks are decided to be the number of pixels since we want to calculate the output of every output-pixel. ANd for simplicity, we make sure that the grid-configuration is such that the number of blocks along each dimension is less than or equal to the number of pixels along each dimension. 

### Kernel Definition
The decimation kernel has the following arguments
1. Pointer to input-image in the global-memory
2. pointer to input-kernel in the global-memory
3. sampling-factor
4. Pointer to the global-memory, allocated for output
5. array containing dimensions of the input-image
6. array containing dimensions of the kernel-matrix
7. number of elements in the kernel
```C
// global functions
__global__ void decimation_ip_cuda(double *d_inputPointerA,
                                   double *d_inputKernel,
                                   const int samplingFactor,
                                   double *d_outputPointer,
                                   mwSize *d_inputDimensionsA,
                                   mwSize *d_inputDimensionsKernel,
                                   mwSize *d_outputDimensions,
                                   const int numElementsKernel)
{
    ...
}
```

<!-- ======================================================================= -->
Since filtering is a collaborative processing, we use shared memory for this. For this code we're using dynamic shared memory, so we declare the pointer we'll be using to point to the shared-memory in the following manner
```C
    // declaring shared memory
    extern __shared__ double sharedMem[];
```

<!-- ======================================================================= -->
Next, we produce mapping from the thread-address to output-index that particular block is responsible for. This is followed by obtaining the linear-index for the same output-index. This is done in the following manner
```C
// Getting coordinates of the final Output
    const int tidx = blockIdx.x;
    const int tidy = blockIdx.y;
    const int tidz = blockIdx.z;
    const int linearIndex2D_Output = \
                tidx + \
                tidy*d_outputDimensions[0] + \
                tidz*d_outputDimensions[0]*d_outputDimensions[1];

```

<!-- ======================================================================= -->
Here, we calculate the input-index the current block is responsible for. 
```C
    // Getting the coordinates of the pixel for which we wanna filter
    const int tidx_input = tidx*samplingFactor;
    const int tidy_input = tidy*samplingFactor;
    const int tidz_input = tidz;
```

<!-- ======================================================================= -->

```C
    // Finding the coordinates of the kernel point
    const int tidx_kernel = threadIdx.x;
    const int tidy_kernel = threadIdx.y;
    const int tidz_kernel = threadIdx.z;
    const int linearIndex2D_kernel = \
        tidx_kernel + \
        tidy_kernel*d_inputDimensionsKernel[0] + \
        tidz_kernel*d_inputDimensionsKernel[0]*d_inputDimensionsKernel[1];
```

<!-- ======================================================================= -->

```C
    // The neighbour pixel this thread is responsible for
    const int tidx_neighbour = tidx_input - tidx_kernel;
    const int tidy_neighbour = tidy_input - tidy_kernel;
    const int tidz_neighbour = tidz_input;
    const int linearIndex2D_neighbour = \
                tidx_neighbour + \
                tidy_neighbour*d_inputDimensionsA[0] + \
                tidz_neighbour*d_inputDimensionsA[0]*d_inputDimensionsA[1];
```

<!-- ======================================================================= -->

```C
    // Finding dot-product
    if(tidx_neighbour<d_inputDimensionsA[0] && \
        tidy_neighbour<d_inputDimensionsA[1] && \
        tidx_kernel<d_inputDimensionsKernel[0] && \
        tidy_kernel<d_inputDimensionsKernel[1])
    {
        // finding the product and writing to shared memory
        sharedMem[linearIndex2D_kernel] = \
            (double)d_inputPointerA[linearIndex2D_neighbour]*\
                d_inputKernel[linearIndex2D_kernel];
    }
    else if(tidx_kernel<d_inputDimensionsKernel[0] && \
            tidy_kernel<d_inputDimensionsKernel[1])
    {
        // setting the invalid values to be zero so that we can 
        // just add without checking the bounds or what nots
        sharedMem[linearIndex2D_kernel] = \
            (double)0;
    }


```

<!-- ======================================================================= -->

```C
    // Getting the first thread to add these values up
    if( threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        // variable that holds the accumulated values 
        double accuSum = (double)0;

        // accumulating values
        for(int i = 0; \
            i<d_inputDimensionsKernel[0]*d_inputDimensionsKernel[1]; \
            ++i)
        {
            accuSum = accuSum + (double)sharedMem[i];
        }

        // writing the values
        if(tidx_input<d_inputDimensionsA[0] && 
           tidy_input<d_inputDimensionsA[1])
            d_outputPointer[linearIndex2D_Output] = (double)accuSum;

    }

    // syncing
    __syncthreads();
```

<!-- ======================================================================= -->
Putting it together the kernel definition should look like this
```C
// global functions
__global__ void decimation_ip_cuda(double *d_inputPointerA,
                                   double *d_inputKernel,
                                   const int samplingFactor,
                                   double *d_outputPointer,
                                   mwSize *d_inputDimensionsA,
                                   mwSize *d_inputDimensionsKernel,
                                   mwSize *d_outputDimensions,
                                   const int numElementsKernel)
{
    // declaring shared memory
    extern __shared__ double sharedMem[];

    // Getting coordinates of the final Output
    const int tidx = blockIdx.x;
    const int tidy = blockIdx.y;
    const int tidz = blockIdx.z;
    const int linearIndex2D_Output = \
                tidx + \
                tidy*d_outputDimensions[0] + \
                tidz*d_outputDimensions[0]*d_outputDimensions[1];

    // Getting the coordinates of the pixel for which we wanna filter
    // const int samplingFactor = (int)d_inputSamplingFactor[0];
    const int tidx_input = tidx*samplingFactor;
    const int tidy_input = tidy*samplingFactor;
    const int tidz_input = tidz;

    // Finding the coordinates of the kernel point
    const int tidx_kernel = threadIdx.x;
    const int tidy_kernel = threadIdx.y;
    const int tidz_kernel = threadIdx.z;
    const int linearIndex2D_kernel = \
        tidx_kernel + \
        tidy_kernel*d_inputDimensionsKernel[0] + \
        tidz_kernel*d_inputDimensionsKernel[0]*d_inputDimensionsKernel[1];

    // The neighbour pixel this thread is responsible for
    const int tidx_neighbour = tidx_input - tidx_kernel;
    const int tidy_neighbour = tidy_input - tidy_kernel;
    const int tidz_neighbour = tidz_input;
    const int linearIndex2D_neighbour = \
                tidx_neighbour + \
                tidy_neighbour*d_inputDimensionsA[0] + \
                tidz_neighbour*d_inputDimensionsA[0]*d_inputDimensionsA[1];
    
    // Finding dot-product
    if(tidx_neighbour<d_inputDimensionsA[0] && \
        tidy_neighbour<d_inputDimensionsA[1] && \
        tidx_kernel<d_inputDimensionsKernel[0] && \
        tidy_kernel<d_inputDimensionsKernel[1])
    {
        // finding the product and writing to shared memory
        sharedMem[linearIndex2D_kernel] = \
            (double)d_inputPointerA[linearIndex2D_neighbour]*\
                d_inputKernel[linearIndex2D_kernel];
    }
    else if(tidx_kernel<d_inputDimensionsKernel[0] && \
            tidy_kernel<d_inputDimensionsKernel[1])
    {
        // setting the invalid values to be zero so that we can 
        // just add without checking the bounds or what nots
        sharedMem[linearIndex2D_kernel] = \
            (double)0;
    }

    // Getting the first thread to add these values up
    if( threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        // variable that holds the accumulated values 
        double accuSum = (double)0;

        // accumulating values
        for(int i = 0; \
            i<d_inputDimensionsKernel[0]*d_inputDimensionsKernel[1]; \
            ++i)
        {
            accuSum = accuSum + (double)sharedMem[i];
        }

        // writing the values
        if(tidx_input<d_inputDimensionsA[0] && 
           tidy_input<d_inputDimensionsA[1])
            d_outputPointer[linearIndex2D_Output] = (double)accuSum;

    }

    // syncing
    __syncthreads();

}
```



<!-- B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B -->
<!-- B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B -->
<!-- B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B -->
<!-- B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B -->
### Mex-Function

In the mex-function, we first start by establishing the validity of the input. We check the number of inputs, the expected number of outputs and the data-type of inputs. This is done in the following manner
```C
// gateway function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // checking number of inputs
    if(nrhs!=3)
        mexErrMsgTxt("Number of Inputs are Wrong \n");

    // checking number of expected outputs
    if(nlhs!=1)
        mexErrMsgTxt("Number of Expected Outputs are Wrong \n");

    // checking input data-types
    if(!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
        mexErrMsgTxt("The first argument is expected to be a matrix of type, double \n");
    if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]))
        mexErrMsgTxt("The second argument is expected to be a matrix of type, double \n");
    if(!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]))
        mexErrMsgTxt("The third argument is expected to be a matrix of type, double \n");
    
    ...
}
```

<!-- ======================================================================= -->
Once the input-data's validity has been established, next we setup the arguments to its appropriate data-structures. The input-image and the kernel-matrix are initialised to objects of the structure, *CustomGPUObject*. Note that we do not use an object of the mentioned class to store the sampling factor because to send scalars to the global-memory, we do not require CUDA API calls. Once the inputs have been setup, we send the input-image and kernel-matrix to the device global-memory using the class method, *copyFromHostDevice()*. 
```C
    // Fetching Inputs
    CustomGPUObject inputImage(prhs[0]);
    CustomGPUObject kernelMatrix(prhs[1]);
    const int samplingFactor = (int)mxGetScalar(prhs[2]);

    // Sending the Data to the Device
    inputImage.copyFromHostToDevice();
    kernelMatrix.copyFromHostToDevice();
```

<!-- ======================================================================= -->
Next, we setup the output-dimensions. Since the output is a decimated image, the output-image will have dimensions of the original image divided by the sampling-factor. 
```C
    // setting up output dimension
    mwSize *outputDimensions = (mwSize *)malloc(inputImage.numDimensions*sizeof(mwSize));
    memcpy(outputDimensions, 
           inputImage.inputDimensions, 
           inputImage.numDimensions*sizeof(mwSize));
    outputDimensions[0] = (mwSize)((inputImage.inputDimensions[0] + \
        samplingFactor - 1)/samplingFactor);
    outputDimensions[1] = (mwSize)((inputImage.inputDimensions[1] + \
        samplingFactor - 1)/samplingFactor);

```

<!-- ======================================================================= -->
Next, we create a Matlab matrix using the function [*mxCreateNumericArray*](https://www.mathworks.com/help/matlab/apiref/mxcreatenumericarray.html?searchHighlight=mxcreatenumericarray&s_tid=srchtitle_support_results_1_mxcreatenumericarray). This created matrix is then used to create an object of our class, *CustomGPUObject*. We use this object instead of the pointer directly to make coding easier. As usual, since our work is on the GPU, we use the class method, *copyFromHostToDevice()* to make memory for the output and sending the output-dimensions to the GPU memory.
```C
    // setting up output
    plhs[0] = mxCreateNumericArray(inputImage.numDimensions,
                                   outputDimensions,
                                   mxDOUBLE_CLASS,
                                   mxREAL);
    CustomGPUObject outputImage(plhs[0]);
    outputImage.copyFromHostToDevice();
```

<!-- ======================================================================= -->
Next,we prepare for the function call by setting up the block-configuration and grid-configuration. The block-configuration and grid-configuration are chosein in such a way that the block-configurations along each-dimensions are greater than and nearest power of 2 of the kernel-dimensions. The grid-configurations are chosen to be the dimensions of the input-image. 

Since we're using shared-memory for this task, the amount must be passed as the thread launch-configuration parameter. The launch-configuration parameter's setup and kernel-launch is shown below
```C
    // kernel-call
    dim3 blockConfiguration(pow2roundup(kernelMatrix.inputDimensions[0]),
                            pow2roundup(kernelMatrix.inputDimensions[1]),
                            1);
    dim3 gridConfiguration((int)outputImage.inputDimensions[0],
                           (int)outputImage.inputDimensions[1],
                            max(1, int(inputImage.inputDimensions[2])));
    decimation_ip_cuda<<<gridConfiguration,
                         blockConfiguration,
                         kernelMatrix.numElements*sizeof(double)>>>
                                (inputImage.d_inputPointer_real,
                                 kernelMatrix.d_inputPointer_real,
                                 samplingFactor,
                                 outputImage.d_inputPointer_real,
                                 inputImage.d_inputDimensions,
                                 kernelMatrix.d_inputDimensions,
                                 outputImage.d_inputDimensions,
                                 kernelMatrix.numElements);
```

<!-- ======================================================================= -->
Once the function call is complted, we copy the output-results from the gpu memory to the host-memory using the method, *copyFromDeviceToHost()* of the class, *CustomGPUObject*.
```C
    // fetching outputs from device global-memory to host-memory
    outputImage.copyFromDeviceToHost();

    // shutting down
    cudaDeviceSynchronize();
    cudaDeviceReset();
```

### Final CUDA 
Putting it all together, we should get the following
```C
/*
    Aim: Implementing Image Decimation Using Custom CUDA-Kernels
*/ 

// header-files
#include "mex.h"
#include "../Beamforming/booktools.h"

// function to find the next power of 2
inline int
pow2roundup (int x)
{
    if (x < 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}

// global functions
__global__ void decimation_ip_cuda(double *d_inputPointerA,
                                   double *d_inputKernel,
                                   const int samplingFactor,
                                   double *d_outputPointer,
                                   mwSize *d_inputDimensionsA,
                                   mwSize *d_inputDimensionsKernel,
                                   mwSize *d_outputDimensions,
                                   const int numElementsKernel)
{
    // declaring shared memory
    extern __shared__ double sharedMem[];

    // Getting coordinates of the final Output
    const int tidx = blockIdx.x;
    const int tidy = blockIdx.y;
    const int tidz = blockIdx.z;
    const int linearIndex2D_Output = \
                tidx + \
                tidy*d_outputDimensions[0] + \
                tidz*d_outputDimensions[0]*d_outputDimensions[1];

    // Getting the coordinates of the pixel for which we wanna filter
    // const int samplingFactor = (int)d_inputSamplingFactor[0];
    const int tidx_input = tidx*samplingFactor;
    const int tidy_input = tidy*samplingFactor;
    const int tidz_input = tidz;

    // Finding the coordinates of the kernel point
    const int tidx_kernel = threadIdx.x;
    const int tidy_kernel = threadIdx.y;
    const int tidz_kernel = threadIdx.z;
    const int linearIndex2D_kernel = \
        tidx_kernel + \
        tidy_kernel*d_inputDimensionsKernel[0] + \
        tidz_kernel*d_inputDimensionsKernel[0]*d_inputDimensionsKernel[1];

    // The neighbour pixel this thread is responsible for
    const int tidx_neighbour = tidx_input - tidx_kernel;
    const int tidy_neighbour = tidy_input - tidy_kernel;
    const int tidz_neighbour = tidz_input;
    const int linearIndex2D_neighbour = \
                tidx_neighbour + \
                tidy_neighbour*d_inputDimensionsA[0] + \
                tidz_neighbour*d_inputDimensionsA[0]*d_inputDimensionsA[1];
    
    // Finding dot-product
    if(tidx_neighbour<d_inputDimensionsA[0] && \
        tidy_neighbour<d_inputDimensionsA[1] && \
        tidx_kernel<d_inputDimensionsKernel[0] && \
        tidy_kernel<d_inputDimensionsKernel[1])
    {
        // finding the product and writing to shared memory
        sharedMem[linearIndex2D_kernel] = \
            (double)d_inputPointerA[linearIndex2D_neighbour]*\
                d_inputKernel[linearIndex2D_kernel];
    }
    else if(tidx_kernel<d_inputDimensionsKernel[0] && \
            tidy_kernel<d_inputDimensionsKernel[1])
    {
        // setting the invalid values to be zero so that we can 
        // just add without checking the bounds or what nots
        sharedMem[linearIndex2D_kernel] = \
            (double)0;
    }

    // Getting the first thread to add these values up
    if( threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        // variable that holds the accumulated values 
        double accuSum = (double)0;

        // accumulating values
        for(int i = 0; \
            i<d_inputDimensionsKernel[0]*d_inputDimensionsKernel[1]; \
            ++i)
        {
            accuSum = accuSum + (double)sharedMem[i];
        }

        // writing the values
        if(tidx_input<d_inputDimensionsA[0] && 
           tidy_input<d_inputDimensionsA[1])
            d_outputPointer[linearIndex2D_Output] = (double)accuSum;

    }

    // syncing
    __syncthreads();

}


// gateway function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // checking number of inputs
    if(nrhs!=3)
        mexErrMsgTxt("Number of Inputs are Wrong \n");

    // checking number of expected outputs
    if(nlhs!=1)
        mexErrMsgTxt("Number of Expected Outputs are Wrong \n");

    // checking input data-types
    if(!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
        mexErrMsgTxt("The first argument is expected to be a matrix of type, double \n");
    if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]))
        mexErrMsgTxt("The second argument is expected to be a matrix of type, double \n");
    if(!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]))
        mexErrMsgTxt("The third argument is expected to be a matrix of type, double \n");

    // Fetching Inputs
    CustomGPUObject inputImage(prhs[0]);
    CustomGPUObject kernelMatrix(prhs[1]);
    const int samplingFactor = (int)mxGetScalar(prhs[2]);

    // Sending the Data to the Device
    inputImage.copyFromHostToDevice();
    kernelMatrix.copyFromHostToDevice();

    // setting up output dimension
    mwSize *outputDimensions = (mwSize *)malloc(inputImage.numDimensions*sizeof(mwSize));
    memcpy(outputDimensions, 
           inputImage.inputDimensions, 
           inputImage.numDimensions*sizeof(mwSize));
    outputDimensions[0] = (mwSize)((inputImage.inputDimensions[0] + \
        samplingFactor - 1)/samplingFactor);
    outputDimensions[1] = (mwSize)((inputImage.inputDimensions[1] + \
        samplingFactor - 1)/samplingFactor);

    // setting up output
    plhs[0] = mxCreateNumericArray(inputImage.numDimensions,
                                   outputDimensions,
                                   mxDOUBLE_CLASS,
                                   mxREAL);
    CustomGPUObject outputImage(plhs[0]);
    outputImage.copyFromHostToDevice();

    // kernel-call
    dim3 blockConfiguration(pow2roundup(kernelMatrix.inputDimensions[0]),
                            pow2roundup(kernelMatrix.inputDimensions[1]),
                            1);
    dim3 gridConfiguration((int)outputImage.inputDimensions[0],
                           (int)outputImage.inputDimensions[1],
                            max(1, int(inputImage.inputDimensions[2])));
    decimation_ip_cuda<<<gridConfiguration,
                         blockConfiguration,
                         kernelMatrix.numElements*sizeof(double)>>>
                                (inputImage.d_inputPointer_real,
                                 kernelMatrix.d_inputPointer_real,
                                 samplingFactor,
                                 outputImage.d_inputPointer_real,
                                 inputImage.d_inputDimensions,
                                 kernelMatrix.d_inputDimensions,
                                 outputImage.d_inputDimensions,
                                 kernelMatrix.numElements);

    // fetching outputs from device global-memory to host-memory
    outputImage.copyFromDeviceToHost();

    // shutting down
    cudaDeviceSynchronize();
    cudaDeviceReset();

}
```

<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
## References
- MathWorks. “Phantom.” MATLAB Documentation, MathWorks, [Link](www.mathworks.com/help/images/ref/phantom.html#d126e261153). Accessed 26 Sept. 2024.
- MathWorks. “mxCreateNumericArray.” MATLAB API Reference, MathWorks, [Link](www.mathworks.com/help/matlab/apiref/mxcreatenumericarray.html?searchHighlight=mxcreatenumericarray&s_tid=srchtitle_support_results_1_mxcreatenumericarray). Accessed 26 Sept. 2024.
- “Parks–McClellan Filter Design Algorithm.” Wikipedia: The Free Encyclopedia, Wikimedia Foundation, 26 Sept. 2024, en.wikipedia.org/wiki/Parks–McClellan_filter_design_algorithm. Accessed 26 Sept. 2024.