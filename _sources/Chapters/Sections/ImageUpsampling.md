# Upsampling (not interpolation)

Image upsampling is a technique used to increase the resolution of an image by inserting additional pixels between the original ones, effectively enlarging the image while attempting to preserve its visual quality. Unlike interpolation, which estimates the values of new pixels based on neighboring pixels, upsampling primarily involves the insertion of zeros or repeated values in a systematic way, followed by filtering to smooth out the result. In mathematical terms, if the original image is represented as a matrix \( I \) of size \( M \times N \), upsampling by a factor of \( L \) involves creating a new matrix \( I_{up} \) of size \( LM \times LN \), where the new matrix has zeros or repeated pixel values inserted between the original ones. For instance, when upsampling by zero insertion, the upsampled image \( I_{up} \) can be represented as:

$$
I_{up}(x, y) = 
\begin{cases} 
    I(x/L, y/L) & \text{if } x = kL \text{ and } y = lL \text{ for integers } k, l \\
    0 & \text{otherwise}
\end{cases}
$$

To achieve a visually pleasing result, this upsampled image is typically processed with a low-pass filter, which interpolates the zero or repeated values to create smooth transitions between the original pixel values.

Image upsampling is widely used in various applications, such as in digital zooming, where an image is magnified without capturing new data, thus requiring the enhancement of existing pixels to maintain quality. In satellite and aerial imaging, upsampling is crucial for enhancing low-resolution images, allowing for detailed analysis of geographical features and urban landscapes. Additionally, upsampling is employed in medical imaging, where higher resolution images can improve the visualization of anatomical structures and aid in diagnosis. For example, in computed tomography (CT) scans, upsampling can help generate more detailed cross-sectional images of the body. Moreover, upsampling plays a significant role in the field of image super-resolution, where machine learning algorithms are trained to upscale images, producing higher resolution outputs from low-resolution inputs, often used in security and surveillance systems to enhance the clarity of captured images.


## Matlab Code
```Matlab
%{
    Aim: Demonstrating Upsampling with CUDA Kernels
%}

%% Basic setup
clc; clear;

%% Compiling Code
mexcuda upsampling_ip_cuda.cu

%% Preparing Input
inputImage = phantom(256);
inputImage = repmat(inputImage, [1,1,3]);
upsamplingFactor = 2;

%% Calling function
outputImage = upsampling_ip_cuda(inputImage, upsamplingFactor);

%% Plotting
figure(1);
subplot(1,2,1); imagesc(inputImage); colorbar; title("Input Image"); 
subplot(1,2,2); imagesc(outputImage); colorbar; title("Output Image");
```

## CUDA Code

```C
/*
    Aim: Implementing Image Upsampling using Custom Cuda kernels
*/ 

// header files
#include "mex.h"
#include "../Beamforming/booktools.h"

// global function
__global__ void upsampling_ip_cuda(double *d_inputPointerA,
                                   const int samplingFactor,
                                   double *d_outputPointer,
                                   const mwSize *d_inputDimensionsA,
                                   mwSize *d_outputDimensions)
{
    // getting thread coordinates
    const int tidx = threadIdx.x + blockIdx.x*blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y*blockDim.y;
    const int tidz = threadIdx.z + blockIdx.z*blockDim.z;

    // getting linear index into the input
    const int linearIndex2D_Input = \
                tidx + \
                tidy*d_inputDimensionsA[0] + \
                tidz*d_inputDimensionsA[0]*d_inputDimensionsA[1];

    // finding the destination coordinates
    const int tidx_output = tidx*samplingFactor;
    const int tidy_output = tidy*samplingFactor;
    const int tidz_output = tidz;

    // getting a linear index to the output
    const int linearIndex2D_Output = \
                tidx_output + \
                tidy_output*d_outputDimensions[0] + \
                tidz_output*d_outputDimensions[0]*d_outputDimensions[1];

    // copying the values to the output
    if(tidx_output<d_outputDimensions[0] && tidy_output<d_outputDimensions[1])
    {
        d_outputPointer[linearIndex2D_Output] = \
            (double)d_inputPointerA[linearIndex2D_Input];
    }

    // syncing
    __syncthreads();

}

// gateway function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // checking number of inputs
    if(nrhs!=2)
        mexErrMsgTxt("Number of Inputs are Wrong \n");
    
    // checking number of outputs
    if(nlhs!=1)
        mexErrMsgTxt("Number of Expected Outputs are Wrong \n");
    
    // checking input data type
    if(!mxIsDouble(prhs[0]) || mxIsComplex(prhs[1]))
        mexErrMsgTxt("Data type of first argument is wrong \n");
    if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]))
        mexErrMsgTxt("Data-type of second argument is wrong \n");

    // setting up inputs
    CustomGPUObject inputImage(prhs[0]);
    const int samplingFactor = (int)mxGetScalar(prhs[1]);
    inputImage.copyFromHostToDevice();
    
    // making output's dimension-array
    mwSize *outputDimensions = (mwSize *)malloc(inputImage.numDimensions*sizeof(mwSize));
    memcpy(outputDimensions, inputImage.inputDimensions, inputImage.numDimensions*sizeof(mwSize));
    outputDimensions[0] = inputImage.inputDimensions[0]*(mwSize)samplingFactor;
    outputDimensions[1] = inputImage.inputDimensions[1]*(mwSize)samplingFactor;

    // Setting up output
    plhs[0] = mxCreateNumericArray(inputImage.numDimensions,
                                   outputDimensions,
                                   mxDOUBLE_CLASS,
                                   mxREAL);
    CustomGPUObject outputImage(plhs[0]);
    outputImage.copyFromHostToDevice();

    // preparing for function call
    dim3 blockConfiguration(32,32,1);
    dim3 gridConfiguration((int)(inputImage.inputDimensions[0]+blockConfiguration.x-1)/blockConfiguration.x,
                           (int)(inputImage.inputDimensions[1]+blockConfiguration.y-1)/blockConfiguration.y,
                            max(1, (int)inputImage.inputDimensions[2]));

    // calling function
    upsampling_ip_cuda<<<gridConfiguration,
                         blockConfiguration>>>(inputImage.d_inputPointer_real,
                                               samplingFactor,
                                               outputImage.d_inputPointer_real,
                                               inputImage.d_inputDimensions,
                                               outputImage.d_inputDimensions);

    // fetching data from device global-memory to host memory
    outputImage.copyFromDeviceToHost();

    // shutting down
    cudaDeviceSynchronize();
    cudaDeviceReset();
}
```
