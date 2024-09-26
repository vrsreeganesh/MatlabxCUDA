# Upsampling

Interpolation is the process by which we increase the size of an image without loss of information and maintaining the visual clarity. Image upsampling is the first step of this process. Image upsampling primarily involves the insertion of a number of zeros (depending on the factor by which interpolation/upsampling is taking place) in between pixels of the original image. Mathematically, if the original image is represented as a matrix, $I$, of size $M \times N$ , upsampling by a factor of $L$  involves creating a new matrix $I_{up}$ of size $LM \times LN$, where the new matrix has zeros or repeated pixel values inserted between the original ones.

$$
I_{up}(x, y) = 
\begin{cases} 
    I(x/L, y/L) & \text{if } x = kL \text{ and } y = lL \text{ for integers } k, l \\
    0 & \text{otherwise}
\end{cases}
$$

Since image upsampling is the first step in classical image interpolation, it will also be used in all the places where interpolation is used. 


## Matlab
In Matlab, we compile the code, produce the two arguments required for this process. This process reuqires two inputs: the image to be upsampled, and the factor by which interpolation must take place.

Before we prepare the arguments and call the function, the first step is to compile the CUDA code so that it produces the executable, which is what we'll be calling from our code. Note that this does **NOT** need to be compiled before the function is to be called. Ideally, the function just needs to be compiled once its completed or any changes has been made. The compiling-line is thrown into the script purely for simplicity and is often a good practice when developing, as any errors would be immediately brought to your attention.

```Matlab
%% Compiling Code
mexcuda upsampling_ip_cuda.cu
```

To create the image argument, feel free to read in any image of your liking (3-channels) and feed it as the argument to the function. In our example, however, we use Matlab's *phantom* function to generate an image. The function, essentially, generates a head-phantom that is often used to test the numerical accuracy of image reconstruction algorithms. Though not relevant to our task, this example uses it because my intention is to get you, the reader, to be able to run the code with minimal external dependencies. And the phantom function is available in the base library of Matlab. So, we stick to *phantom*. If this function piques your interest, feel free to read up more at [phantom Matlab Documentation](https://www.mathworks.com/help/images/ref/phantom.html#d126e261153).

```Matlab
%% Preparing Input
inputImage = phantom(256);
inputImage = repmat(inputImage, [1,1,3]);
```

The function takes in multiple arguments but for our context, all we need to know is that passing a non-zero integer, $N$, to *phantom()* makes the function return a phantom image of dimensions $N \times N$. For the demonstration, we choose the second argument, the upsampling factor, to be, $2$. 

```Matlab
%% Preparing Second Input
upsamplingFactor = 2;
```

Now that we have both the arguments, we call the function. The function returns the upsampled image. 

```Matlab
%% Calling function
outputImage = upsampling_ip_cuda(inputImage, upsamplingFactor);
```

This image is displayed side-by-side so that we can see the difference in dimensions. 
```Matlab
%% Plotting
figure(1);
subplot(1,2,1); imagesc(inputImage); colorbar; title("Input Image"); 
subplot(1,2,2); imagesc(outputImage); colorbar; title("Output Image");
```



### Full Matlab Code

```Matlab
%{
    Aim: Demonstrating Upsampling with CUDA Kernels
%}

%% Basic setup
clc; clear;

%% Compiling Code
mexcuda upsampling_ip_cuda.cu

%% Preparing First Input
inputImage = phantom(256);
inputImage = repmat(inputImage, [1,1,3]);

%% Preparing Second Input
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
