# Downsampling

Image downsampling is the process of reducing the resolution of an image by decreasing its number of pixels. This is typically achieved by averaging or combining groups of pixels into single pixels. A common method for downsampling is using a box filter, where a block of \( n \times n \) pixels in the original image is averaged to produce a single pixel in the downsampled image. Mathematically, if the original image has pixel values \( I(x, y) \), the downsampled image with a scaling factor of \( n \) will have pixel values given by:
\[ I_{\text{downsampled}}(i, j) = \frac{1}{n^2} \sum_{k=0}^{n-1} \sum_{l=0}^{n-1} I(i \cdot n + k, j \cdot n + l) \]
where \( i \) and \( j \) are the coordinates in the downsampled image. This process effectively reduces the spatial resolution, making the image smaller in size and often smoother.


Downsampling is widely used in various applications where reduced image resolution is sufficient or desirable. In digital photography, downsampling is employed to create thumbnail images for quick previews. It is also used in image compression techniques to reduce the amount of data required to represent an image, making storage and transmission more efficient. Another application is in multi-scale image analysis, such as in object detection and recognition tasks, where images at different resolutions are analyzed to capture features at various scales. For example, in medical imaging, downsampling can be used to quickly review large sets of images before examining them in full detail. In each of these cases, downsampling helps manage data more efficiently without significantly compromising the essential information in the image.


## Matlab Code

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
