# RGB to YCbCr

Converting RGB(Red, Green, Blue) to YCbCr(Luminance, Blue-difference Chrominance, Red-difference Chrominance) is a common process in image and video compression. The RGB color format/model is often used for display purposes because it aligns well with the way the human eye perceives colours. 

However, RGB is not very efficient for certain tasks such as compressing images and videos because it does not separate the luminance information(brightness) from the chrominance information (colour), which is crucial for reducing data-size without significantly affecting visual quality. The YCbCR model separates thhe image into a luma component, *Y*,  and two chroma components (*Cb* and *Cr*), making it more suitable for compression algorithms used in standards like JPEG, MPEG and H.264. 

So converting from one format to the other is rather important. Mathematically, the conversion is done in the following manner.

$$
Y = 0.299R + 0.587G + 0.114B \\
Cb = 128 - 0.168736R - 0.331264G + 0.5B \\
Cr = 128 + 0.5R - 0.418688G - 0.081312B	
$$


In these equations, the coefficients reflect the contribution of each RGB component to the Y, Cb, and Cr values, and the addition of 128 centers the chroma components around 128 to allow for both positive and negative values within an 8-bit range (0-255). This separation allows for more efficient compression because the human eye is more sensitive to changes in luminance than in chrominance. As a result, chrominance components can be subsampled, reducing the amount of data needed without perceptible loss of quality. This principle is widely used in various applications, including digital television, DVDs, and many streaming services, where efficient compression is essential for storage and transmission.

<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->

## MATLAB Code

<!-- ======================================================================= -->
Before we prepare the arguments and call the function, the first step is to compile the CUDA code so that it produces the executable, which is what we'll be calling from our code. Note that this does **NOT** need to be compiled before the function is to be called. Ideally, the function just needs to be compiled once its completed or any changes has been made. The compiling-line is thrown into the script purely for simplicity and is often a good practice when developing, as any errors would be immediately brought to your attention.
```Matlab
%% Compiling
mexcuda RGB2YCbCr_cuda.cu
```

<!-- ======================================================================= -->
To create the image argument, feel free to read in any image of your liking (3-channels) and feed it as the argument to the function. In our example, however, we use Matlab's *phantom* function to generate an image. The function, essentially, generates a head-phantom that is often used to test the numerical accuracy of image reconstruction algorithms. Though not relevant to our task, this example uses it because my intention is to get you, the reader, to be able to run the code with minimal external dependencies. And the phantom function is available in the base library of Matlab. So, we stick to *phantom*. If this function piques your interest, feel free to read up more at [phantom Matlab Documentation](https://www.mathworks.com/help/images/ref/phantom.html#d126e261153). 

```Matlab
%% Preparing Arguments
inputImage = phantom(256);
inputImage = repmat(inputImage, [1,1,3]);
```
<!-- ======================================================================= -->
Next, we call the Matlab's built-in function that converts RGB to YCbCr, [*rgb2ycbcr*](www.mathworks.com/help/images/ref/rgb2ycbcr.html). This function helps us verify the output of our method. So we also call this function. 
```Matlab
matlabYCbCr = rgb2ycbcr(inputImage);
matlabRGB = ycbcr2rgb(matlabYCbCr);
```
<!-- ======================================================================= -->
Next, we call the function with the prepared/read input in the following manner.
```Matlab
%% Calling the Function
outputImage = RGB2YCbCr_cuda(inputImage);
```
<!-- ======================================================================= -->
Once the output has been created, we use another Matlab function, [*ycbcr2rgb*](https://www.mathworks.com/help/images/ref/ycbcr2rgb.html). This is also used to verify the output of our function. 
```Matlab
%% Converting the image back to RGB to check if we've done a good job
rgbOutputImage = ycbcr2rgb(outputImage);
```
<!-- ======================================================================= -->
Finally, we visualize the output and other plots relevant to verify the output. 
```Matlab
%% Plotting the Image
figure(1);
subplot(2,3,1); imshow(outputImage(:,:,1)); title("Y-cuda");
subplot(2,3,2); imshow(outputImage(:,:,2)); title("Cb-cuda");
subplot(2,3,3); imshow(outputImage(:,:,3)); title("Cr-cuda");

subplot(2,3,4); imshow(matlabYCbCr(:,:,1)); title("Y-matlab");
subplot(2,3,5); imshow(matlabYCbCr(:,:,2)); title("Cb-matlab");
subplot(2,3,6); imshow(matlabYCbCr(:,:,3)); title("Cr-matlab");
```


### Final Matlab Code
Putting it together, we get the following
```Matlab
%{
    Aim: RGB to YCbCr
%}

%% Basic Setup
clc; clear; close all;

%% Compiling
mexcuda RGB2YCbCr_cuda.cu

%% Preparing Arguments
inputImage = phantom(256);
inputImage = repmat(inputImage, [1,1,3]);

matlabYCbCr = rgb2ycbcr(inputImage);
matlabRGB = ycbcr2rgb(matlabYCbCr);

%% Calling the Function
outputImage = RGB2YCbCr_cuda(inputImage);

%% Converting the image back to RGB to check if we've done a good job
rgbOutputImage = ycbcr2rgb(outputImage);

%% Plotting the Image
figure(1);
subplot(2,3,1); imshow(outputImage(:,:,1)); title("Y-cuda");
subplot(2,3,2); imshow(outputImage(:,:,2)); title("Cb-cuda");
subplot(2,3,3); imshow(outputImage(:,:,3)); title("Cr-cuda");

subplot(2,3,4); imshow(matlabYCbCr(:,:,1)); title("Y-matlab");
subplot(2,3,5); imshow(matlabYCbCr(:,:,2)); title("Cb-matlab");
subplot(2,3,6); imshow(matlabYCbCr(:,:,3)); title("Cr-matlab");
```









<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A -->

## CUDA 
For the CUDA part, we need to define two different function bodies: the kernel-definition and the entry-gateway function. 

### Kernel-Function
The kernel function takes in the following set of arguments. 
<div style="margin-top: -3mm;"></div>


1. The pointer to the input-matrix in the GPU global-memory. 
2. The pointer to the space allocated for the output in the GPU global-memory. 
3. The pointer to the array containing the dimensions of the input-matrix

```C
// global kernels
__global__ void RGB2YCbCr_cuda(double *d_inputPointerA,
                               double *d_outputPointer,
                               mwSize *d_inputDimensionsA)
{
    ...
}
```

<!-- ======================================================================= -->
Next, we produce the mapping from the thread-address to the linear-index of the input-matrix. 
```C
    // Addressing the pixels
    const int tidx = threadIdx.x + blockIdx.x*blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y*blockDim.y;
    const int linearIndex2D = tidx + \
                              tidy*d_inputDimensionsA[0];
```

<!-- ======================================================================= -->
Next, we calculate the three different components. We use the expressions presented before to calculate this. Before we do that, we check the validity of the threads to ensure that we're not performing processing on memory-locations that are invalid. 
```C
// Checking Validity
    if(tidx<d_inputDimensionsA[0] && tidy<d_inputDimensionsA[1])
    {
        // Calculating Y
        d_outputPointer[linearIndex2D] = \
            (double)(0.2126)*d_inputPointerA[linearIndex2D] + \
            (double)(0.7152)*d_inputPointerA[linearIndex2D + d_inputDimensionsA[0]*d_inputDimensionsA[1]] + \
            (double)(0.0722)*d_inputPointerA[linearIndex2D + 2*d_inputDimensionsA[0]*d_inputDimensionsA[1]];

        // Calculating Cb
        d_outputPointer[linearIndex2D + d_inputDimensionsA[0]*d_inputDimensionsA[1]] = \
            (double)(-0.1146)*d_inputPointerA[linearIndex2D] + \
            (double)(-0.3854)*d_inputPointerA[linearIndex2D + d_inputDimensionsA[0]*d_inputDimensionsA[1]] + \
            (double)(0.5)*d_inputPointerA[linearIndex2D + 2*d_inputDimensionsA[0]*d_inputDimensionsA[1]];

        // Calculating Cr
        d_outputPointer[linearIndex2D + 2*d_inputDimensionsA[0]*d_inputDimensionsA[1]] = \
            (double)(0.5)*d_inputPointerA[linearIndex2D] + \
            (double)(-0.4542)*d_inputPointerA[linearIndex2D + d_inputDimensionsA[0]*d_inputDimensionsA[1]] + \
            (double)(-0.0458)*d_inputPointerA[linearIndex2D + 2*d_inputDimensionsA[0]*d_inputDimensionsA[1]];
    }
```

<!-- ======================================================================= -->
Putting it together, the kernel definition must look like the following. 
```C
// global kernels
__global__ void RGB2YCbCr_cuda(double *d_inputPointerA,
                               double *d_outputPointer,
                               mwSize *d_inputDimensionsA)
{
    // Addressing the pixels
    const int tidx = threadIdx.x + blockIdx.x*blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y*blockDim.y;
    const int linearIndex2D = tidx + \
                              tidy*d_inputDimensionsA[0];

    // Checking Validity
    if(tidx<d_inputDimensionsA[0] && tidy<d_inputDimensionsA[1])
    {
        // Calculating Y
        d_outputPointer[linearIndex2D] = \
            (double)(0.2126)*d_inputPointerA[linearIndex2D] + \
            (double)(0.7152)*d_inputPointerA[linearIndex2D + d_inputDimensionsA[0]*d_inputDimensionsA[1]] + \
            (double)(0.0722)*d_inputPointerA[linearIndex2D + 2*d_inputDimensionsA[0]*d_inputDimensionsA[1]];

        // Calculating Cb
        d_outputPointer[linearIndex2D + d_inputDimensionsA[0]*d_inputDimensionsA[1]] = \
            (double)(-0.1146)*d_inputPointerA[linearIndex2D] + \
            (double)(-0.3854)*d_inputPointerA[linearIndex2D + d_inputDimensionsA[0]*d_inputDimensionsA[1]] + \
            (double)(0.5)*d_inputPointerA[linearIndex2D + 2*d_inputDimensionsA[0]*d_inputDimensionsA[1]];

        // Calculating Cr
        d_outputPointer[linearIndex2D + 2*d_inputDimensionsA[0]*d_inputDimensionsA[1]] = \
            (double)(0.5)*d_inputPointerA[linearIndex2D] + \
            (double)(-0.4542)*d_inputPointerA[linearIndex2D + d_inputDimensionsA[0]*d_inputDimensionsA[1]] + \
            (double)(-0.0458)*d_inputPointerA[linearIndex2D + 2*d_inputDimensionsA[0]*d_inputDimensionsA[1]];
    }

    // syncing
    __syncthreads();
}
```

<!-- C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C -->
<!-- C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C -->
<!-- C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C -->
<!-- C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C -->
### Gateway-Function

In the gateway function, we first check the validity of the arguments that are passed from the Matlab script that called this kernel. We check the number of inputs, the expected number of outputs and the data-type of the input. 

```C
// main function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // checking number of inputs
    if(nrhs!=1)
        mexErrMsgTxt("Number of Inputs are Wrong \n");
    // checking number of outputs
    if(nlhs!=1)
        mexErrMsgTxt("Number of Expected Outputs are Wrong \n");
    // checking input data type
    if(!mxIsDouble(prhs[0]))
        mexErrMsgTxt("The Input Data Should be of type, double \n");
    
    ...
}
```

<!-- ======================================================================= -->
Once the inputs have been checked and verified to be valid, we use an object of the class, *CustomGPUObject* to encapsulate the input. We then make the data-structure available in the GPU global-memory space. 
```C
    // Fetching Inputs
    CustomGPUObject inputImage(prhs[0]);
    inputImage.copyFromHostToDevice();
```

<!-- ======================================================================= -->
Next, we create a Matlab matrix that will be used to store the output. So the first step is to create a matrix, using [*mxCreateNumericArray*](https://www.mathworks.com/help/matlab/apiref/mxcreatenumericarray.html?searchHighlight=mxcreatenumericarray&s_tid=srchtitle_support_results_1_mxcreatenumericarray), that has the same dimensions of that of the input since the conversion form RGB to YCbCr doesn't change the dimensionality of the matrix. The created matrix is then encapsulated with the help of an object of the class, *CustomGPUObject*. This allows easier control of the data-structure throguh the class methods. Once this has been allocated, call the class method, *copyFromHostToDevice()*. This function allocates space for the output in the GPU global-memory. 
```C
    // Creating output
    plhs[0] = mxCreateNumericArray(inputImage.numDimensions,
                                   inputImage.inputDimensions,
                                   mxDOUBLE_CLASS,
                                   mxREAL);
    CustomGPUObject outputImage(plhs[0]);
    outputImage.copyFromHostToDevice();
```

<!-- ======================================================================= -->
Once the output has been created, we setup the launch-configuration parameters. The task of converting an image from RGB to YCbCr has a straight-forward mapping for the threads. THis means that the launnch-configuration is also rather straight-forward. We fix the block size to be *(32,32)* and the grid-configuration is chosen in such a way that the number of threads along each dimension is greater than or equal to the number of pixels along each-dimension. Once the launch-configuration parameters have been setup, we launch the kernels into the default stream with the launch-configuration parameters and the function arguments in the following manner. 
```C
    // Prepping for function call
    dim3 blockConfiguration(32,32,1);
    dim3 gridConfiguration((inputImage.inputDimensions[0] + blockConfiguration.x - 1)/blockConfiguration.x,
                           (inputImage.inputDimensions[1] + blockConfiguration.y - 1)/blockConfiguration.y,
                            max(1, (int)inputImage.inputDimensions[2]));
    RGB2YCbCr_cuda<<<gridConfiguration,
                     blockConfiguration>>>(inputImage.d_inputPointer_real,
                                           outputImage.d_inputPointer_real,
                                           inputImage.d_inputDimensions);
```

<!-- ======================================================================= -->
After the function call, the results of the outputs are made available in the host-memory using the class method, *copyFromDeviceToHost()*.
```C
    // Getting data back from device global-memory to host-memory
    outputImage.copyFromDeviceToHost();

    // shutting
    cudaDeviceSynchronize();
    cudaDeviceReset();
```

### Final CUDA Code
<!-- ======================================================================= -->
Putting together the kernel-definition and gateway-function, we obtain the following. 

```C
/*
    Aim: Creating Custom CUDA Kernels converting RGB to YCbCr
*/ 

// header-file
#include "mex.h"
#include "../Beamforming/booktools.h"

// global kernels
__global__ void RGB2YCbCr_cuda(double *d_inputPointerA,
                               double *d_outputPointer,
                               mwSize *d_inputDimensionsA)
{
    // Addressing the pixels
    const int tidx = threadIdx.x + blockIdx.x*blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y*blockDim.y;
    const int linearIndex2D = tidx + \
                              tidy*d_inputDimensionsA[0];

    // Checking Validity
    if(tidx<d_inputDimensionsA[0] && tidy<d_inputDimensionsA[1])
    {
        // Calculating Y
        d_outputPointer[linearIndex2D] = \
            (double)(0.2126)*d_inputPointerA[linearIndex2D] + \
            (double)(0.7152)*d_inputPointerA[linearIndex2D + d_inputDimensionsA[0]*d_inputDimensionsA[1]] + \
            (double)(0.0722)*d_inputPointerA[linearIndex2D + 2*d_inputDimensionsA[0]*d_inputDimensionsA[1]];

        // Calculating Cb
        d_outputPointer[linearIndex2D + d_inputDimensionsA[0]*d_inputDimensionsA[1]] = \
            (double)(-0.1146)*d_inputPointerA[linearIndex2D] + \
            (double)(-0.3854)*d_inputPointerA[linearIndex2D + d_inputDimensionsA[0]*d_inputDimensionsA[1]] + \
            (double)(0.5)*d_inputPointerA[linearIndex2D + 2*d_inputDimensionsA[0]*d_inputDimensionsA[1]];

        // Calculating Cr
        d_outputPointer[linearIndex2D + 2*d_inputDimensionsA[0]*d_inputDimensionsA[1]] = \
            (double)(0.5)*d_inputPointerA[linearIndex2D] + \
            (double)(-0.4542)*d_inputPointerA[linearIndex2D + d_inputDimensionsA[0]*d_inputDimensionsA[1]] + \
            (double)(-0.0458)*d_inputPointerA[linearIndex2D + 2*d_inputDimensionsA[0]*d_inputDimensionsA[1]];
    }

    // syncing
    __syncthreads();

}

// main function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // checking number of inputs
    if(nrhs!=1)
        mexErrMsgTxt("Number of Inputs are Wrong \n");
    // checking number of outputs
    if(nlhs!=1)
        mexErrMsgTxt("Number of Expected Outputs are Wrong \n");
    // checking input data type
    if(!mxIsDouble(prhs[0]))
        mexErrMsgTxt("The Input Data Should be of type, double \n");

    // Fetching Inputs
    CustomGPUObject inputImage(prhs[0]);
    inputImage.copyFromHostToDevice();

    // Creating output
    plhs[0] = mxCreateNumericArray(inputImage.numDimensions,
                                   inputImage.inputDimensions,
                                   mxDOUBLE_CLASS,
                                   mxREAL);
    CustomGPUObject outputImage(plhs[0]);
    outputImage.copyFromHostToDevice();

    // Prepping for function call
    dim3 blockConfiguration(32,32,1);
    dim3 gridConfiguration((inputImage.inputDimensions[0] + blockConfiguration.x - 1)/blockConfiguration.x,
                           (inputImage.inputDimensions[1] + blockConfiguration.y - 1)/blockConfiguration.y,
                            max(1, (int)inputImage.inputDimensions[2]));
    RGB2YCbCr_cuda<<<gridConfiguration,
                     blockConfiguration>>>(inputImage.d_inputPointer_real,
                                           outputImage.d_inputPointer_real,
                                           inputImage.d_inputDimensions);

    // Getting data back from device global-memory to host-memory
    outputImage.copyFromDeviceToHost();

    // shutting
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
- MathWorks. “rgb2ycbcr.” MATLAB Documentation, MathWorks, [Link](www.mathworks.com/help/images/ref/rgb2ycbcr.html). Accessed 26 Sept. 2024.
- MathWorks. “ycbcr2rgb.” MATLAB Documentation, MathWorks, [Link](www.mathworks.com/help/images/ref/ycbcr2rgb.html). Accessed 26 Sept. 2024.
- MathWorks. “mxCreateNumericArray.” MATLAB API Reference, MathWorks, [Link](www.mathworks.com/help/matlab/apiref/mxcreatenumericarray.html?searchHighlight=mxcreatenumericarray&s_tid=srchtitle_support_results_1_mxcreatenumericarray). Accessed 26 Sept. 2024.









