# YCbCr to RGB

Converting YCbCr (Luminance, Blue-difference Chrominance, Red-difference Chrominance) back to RGB (Red, Green, Blue) is a crucial process in digital imaging and video playback. Since many compression algorithms and broadcast standards use YCbCr for efficient data storage and transmission, displaying the images or videos on screens requires converting them back to RGB, the native color space for most displays. The conversion ensures that the colors are accurately represented on devices like monitors, TVs, and mobile screens, which inherently operate using the RGB color model. The conversion from YCbCr to RGB can be expressed with the following equations:

$$
R &= Y + 1.402(Cr - 128) \\
G &= Y - 0.344136(Cb - 128) - 0.714136(Cr - 128) \\
B &= Y + 1.772(Cb - 128)
$$

These equations reverse the process of converting RGB to YCbCr, using the luma (Y) and chroma (Cb and Cr) values to reconstruct the original RGB components. The constants in these equations are derived from the standard YCbCr to RGB transformation matrix, ensuring the correct balance and range of color intensities. This conversion is essential in various applications, including video playback devices, image editing software, and broadcasting systems. For instance, when a video file stored in a compressed format (like H.264) is played on a TV, the YCbCr data is converted to RGB in real-time to render the image accurately on the screen. Similarly, photo editing tools often convert images from YCbCr (used in JPEG) to RGB for accurate color representation during editing.

## Matlab Code
```matlab
%{
    Aim: RGB to YCbCr
%}

%% Basic Setup
clc; clear; close all;

%% Compiling
mexcuda YCbCr2RGB_cuda.cu

%% Preparing Arguments
inputImage = phantom(256);
inputImage = repmat(inputImage, [1,1,3]);
inputImage = rgb2ycbcr(inputImage);

%% Calling a Function
outputImage = YCbCr2RGB_cuda(inputImage);
outputImage_matlab = ycbcr2rgb(inputImage);

%% Plotting
figure(2);
subplot(2,3,1); imshow(outputImage(:,:,1)); title("R-cuda");
subplot(2,3,2); imshow(outputImage(:,:,2)); title("G-cuda");
subplot(2,3,3); imshow(outputImage(:,:,3)); title("B-cuda");

subplot(2,3,4); imshow(outputImage_matlab(:,:,1)); title("R-matlab");
subplot(2,3,5); imshow(outputImage_matlab(:,:,2)); title("G-matlab");
subplot(2,3,6); imshow(outputImage_matlab(:,:,3)); title("B-matlab");    
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
__global__ void YCbCr2RGB_cuda(double *d_inputPointerA,
                               double *d_outputPointer,
                               mwSize *d_inputDimensionsA)
{
    // Addressing the pixels
    const int tidx = threadIdx.x + blockIdx.x*blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y*blockDim.y;
    const int linearIndex2D = \
                tidx + \
                tidy*d_inputDimensionsA[0];

    // Checking Validity
    if(tidx<d_inputDimensionsA[0] && tidy<d_inputDimensionsA[1])
    {
        // Calculating R
        d_outputPointer[linearIndex2D] = \
            (double)(1.0)*d_inputPointerA[linearIndex2D] + \
            (double)(0.0)*d_inputPointerA[linearIndex2D + d_inputDimensionsA[0]*d_inputDimensionsA[1]] + \
            (double)(1.5748)*d_inputPointerA[linearIndex2D + 2*d_inputDimensionsA[0]*d_inputDimensionsA[1]];

        // Calculating G
        d_outputPointer[linearIndex2D + d_inputDimensionsA[0]*d_inputDimensionsA[1]] = \
            0.5 + \
            (double)(1.0)*d_inputPointerA[linearIndex2D] + \
            (double)(-0.1873)*d_inputPointerA[linearIndex2D + d_inputDimensionsA[0]*d_inputDimensionsA[1]] + \
            (double)(-0.4681)*d_inputPointerA[linearIndex2D + 2*d_inputDimensionsA[0]*d_inputDimensionsA[1]];

        // Calculating B
        d_outputPointer[linearIndex2D + 2*d_inputDimensionsA[0]*d_inputDimensionsA[1]] = \
            -0.5 + \
            (double)(1.0)*d_inputPointerA[linearIndex2D] + \
            (double)(1.8556)*d_inputPointerA[linearIndex2D + d_inputDimensionsA[0]*d_inputDimensionsA[1]] + \
            (double)(0)*d_inputPointerA[linearIndex2D + 2*d_inputDimensionsA[0]*d_inputDimensionsA[1]];
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

    // Fetching the Inputs
    CustomGPUObject inputImage(prhs[0]);
    inputImage.copyFromHostToDevice();

    // Creating output
    plhs[0] = mxCreateNumericArray(inputImage.numDimensions,
                                   inputImage.inputDimensions,
                                   mxDOUBLE_CLASS,
                                   mxREAL);
    CustomGPUObject outputImage(plhs[0]);

    // Prepping for function call
    dim3 blockConfiguration(32,32,1);
    dim3 gridConfiguration\
        ((inputImage.inputDimensions[0] + blockConfiguration.x - 1)/blockConfiguration.x,
         (inputImage.inputDimensions[1] + blockConfiguration.y - 1)/blockConfiguration.y,
          max(1, (int)inputImage.inputDimensions[2]));
    YCbCr2RGB_cuda<<<gridConfiguration,
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

