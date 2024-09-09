# RGB to YCbCr

Converting RGB (Red, Green, Blue) to YCbCr (Luminance, Blue-difference Chrominance, Red-difference Chrominance) is a common process in image and video compression. The RGB color model is often used for display purposes because it aligns well with the way human eyes perceive colors. However, RGB is not very efficient for compressing images and videos because it does not separate the luminance information (brightness) from the chrominance information (color), which is crucial for reducing data size without significantly affecting visual quality. The YCbCr model separates the image into a luma component (Y) and two chroma components (Cb and Cr), making it more suitable for compression algorithms used in standards like JPEG, MPEG, and H.264. The conversion from RGB to YCbCr can be expressed using the following equations:
$$
	Y &= 0.299R + 0.587G + 0.114B \\
	Cb &= 128 - 0.168736R - 0.331264G + 0.5B \\
	Cr &= 128 + 0.5R - 0.418688G - 0.081312B	
$$

In these equations, the coefficients reflect the contribution of each RGB component to the Y, Cb, and Cr values, and the addition of 128 centers the chroma components around 128 to allow for both positive and negative values within an 8-bit range (0-255). This separation allows for more efficient compression because the human eye is more sensitive to changes in luminance than in chrominance. As a result, chrominance components can be subsampled, reducing the amount of data needed without perceptible loss of quality. This principle is widely used in various applications, including digital television, DVDs, and many streaming services, where efficient compression is essential for storage and transmission.


## MATLAB Code

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










## CUDA Code
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
