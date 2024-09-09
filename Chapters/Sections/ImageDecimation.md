# Decimation
Image decimation is a process in digital image processing where the resolution of an image is reduced by selecting and retaining a subset of the original image's pixels while discarding the rest. Unlike downsampling, which typically involves averaging groups of pixels to reduce image size, decimation straightforwardly removes pixels based on a specific pattern or algorithm. For instance, if we decimate an image by a factor of 2, we might keep every second pixel in both the horizontal and vertical directions, resulting in an image that retains only a quarter of the original pixels. Mathematically, if \(I(x, y)\) represents the original image, the decimated image \(I_d(x', y')\) can be expressed as \(I_d(x', y') = I(sx', sy')\), where \(s\) is the decimation factor, and \(x'\) and \(y'\) are the coordinates in the decimated image.

An essential step in the decimation process is applying an anti-aliasing filter before removing pixels. Without this step, high-frequency information in the image could cause aliasing, where different signals become indistinguishable, leading to artifacts in the decimated image. The anti-aliasing filter is typically a low-pass filter that smooths the image by reducing high-frequency components. Mathematically, if \(H(f)\) represents the filter transfer function and \(F(f)\) the Fourier transform of the image, the filtered image \(I_f(x, y)\) is given by \(I_f(x, y) = \mathcal{F}^{-1}(H(f) \cdot F(f))\), where \(\mathcal{F}^{-1}\) denotes the inverse Fourier transform. Decimation is used in various contexts such as video processing, where reducing the resolution of frames lowers the bandwidth needed for streaming, and in remote sensing, where it manages the vast data collected by satellite sensors for quicker analysis. In machine learning, particularly in image preprocessing for training models, decimation reduces the input image size, thereby speeding up the training process by lowering computational overhead while preserving essential features for recognition tasks.



## MATLAB Code

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



## CUDA Code

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
