# YCbCr to RGB

Converting images from the YCbCr (Luminance, Blue-difference Chrominance, Red-difference Chrominance) to RGB (Red, Green, Blue) is a crucial transformation in digital image and video-processing. Since many compression algorithms and broadcast standards use *YCbCr* for efficient data-storage and transmission, displaying the images or videos on screens require converting back to the RGB format. Mathematically, this conversion is performed in the following manner. 

$$
R = Y + 1.402(Cr - 128) \\
G = Y - 0.344136(Cb - 128) - 0.714136(Cr - 128) \\
B = Y + 1.772(Cb - 128)
$$


<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
## Matlab Code
In the Matlab code, as usual, we proceed to do the following steps
<div style="margin-top: -3mm;"></div>

1. Compile CUDA code
2. Setup the arguments
3. Function call
4. Exhibit results

```C
%% Compiling
mexcuda YCbCr2RGB_cuda.cu
```

<!-- ======================================================================= -->
To create the image argument, feel free to read in any image of your liking (3-channels) and feed it as the argument to the function. In our example, however, we use Matlab's *phantom* function to generate an image. The function, essentially, generates a head-phantom that is often used to test the numerical accuracy of image reconstruction algorithms. Though not relevant to our task, this example uses it because my intention is to get you, the reader, to be able to run the code with minimal external dependencies. And the phantom function is available in the base library of Matlab. So, we stick to *phantom*. If this function piques your interest, feel free to read up more at [phantom Matlab Documentation](https://www.mathworks.com/help/images/ref/phantom.html#d126e261153). Once this image has been obtained, we create the *YCbCr* version of it using Matlab's function, [*rgb2ycbcr*](www.mathworks.com/help/images/ref/rgb2ycbcr.html). The output obtained from this function is then used as argument to our function. Note that we also pass this to the Matlab's inbuilt function, [*ycbcr2rgb*](www.mathworks.com/help/images/ref/ycbcr2rgb.html) to compare the results obtained from our function vs theirs. 

```C
%% Preparing Arguments
inputImage = phantom(256);
inputImage = repmat(inputImage, [1,1,3]);
inputImage = rgb2ycbcr(inputImage);

%% Calling a Function
outputImage = YCbCr2RGB_cuda(inputImage);
outputImage_matlab = ycbcr2rgb(inputImage);
```
<!-- ======================================================================= -->
The results obtained form all this procedure is then presented in the following manner to compare them against each other. 

```C
%% Plotting
figure(2);
subplot(2,3,1); imshow(outputImage(:,:,1)); title("R-cuda");
subplot(2,3,2); imshow(outputImage(:,:,2)); title("G-cuda");
subplot(2,3,3); imshow(outputImage(:,:,3)); title("B-cuda");

subplot(2,3,4); imshow(outputImage_matlab(:,:,1)); title("R-matlab");
subplot(2,3,5); imshow(outputImage_matlab(:,:,2)); title("G-matlab");
subplot(2,3,6); imshow(outputImage_matlab(:,:,3)); title("B-matlab");    
```

<!-- ======================================================================= -->
### Final Matlab Code
Putting this all together, the final Matlab code should look like this. 


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


<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->

## CUDA 
The CUDA code requires us to define two functions. The first is the gateway function, which is the function that Matlab calls. The second is the global-function that contains the kernel definition. We first start with the kernel-definition. 

### Kernel Definition
<!-- ======================================================================= -->
The kernel takes in three arguments
<div style="margin-top: -3mm;"></div>

1. The pointer to the input-matrix stored in the GPU global-memory
2. The pointer to the memory-allocated for the output in the GPU global-memory
3. The pointer to the input-dimensions stored in the GPU global-memory

```C
// global kernels
__global__ void YCbCr2RGB_cuda(double *d_inputPointerA,
                               double *d_outputPointer,
                               mwSize *d_inputDimensionsA)
{
    ...
}
```

<!-- ======================================================================= -->
Next, we produce mappings from the thread-address to the linear-index of the input-structure. Since this transformation doesn't change the matrix dimensions and only depends on the same pixel at three different channels, the mapping is rather straight-forward.
```C
    // Addressing the pixels
    const int tidx = threadIdx.x + blockIdx.x*blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y*blockDim.y;
    const int linearIndex2D = \
                tidx + \
                tidy*d_inputDimensionsA[0];
```

<!-- ======================================================================= -->
Next, we check the validity of the pixels that the threads are responsible for. This is important so that we do not process, read or write memory that is not part of the data that we need to process. This also has a non-trivial probbaility of happening because there is a good chance that there is more threads allocated than there is required, for optimal performance reasons. After the check, we produce the appropriate channel values using the transformation functions presented in the above section. 
```C
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
```

<!-- ======================================================================= -->
Putting it all together, the kernel function-definition must look like the following. 
```C
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
```

### Gateway Function
In the gateway function, the first step is to check the validity of the inputs. This includes checking the number of inputs, the expected number of outputs and the data-type of the inputs. This is done in the following manner
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
Once the inputs have been verified, we encapulate the inputs using an object of our class, *CustomGPUObject* in the following manner.
```C
    // Fetching the Inputs
    CustomGPUObject inputImage(prhs[0]);
    inputImage.copyFromHostToDevice();
```

<!-- ======================================================================= -->
Next, we create a Matlab matrix of the same dimensions using [*mxCreateNumericArray*](www.mathworks.com/help/matlab/apiref/mxcreatenumericarray.html?searchHighlight=mxcreatenumericarray&s_tid=srchtitle_support_results_1_mxcreatenumericarray). The matrix produced using this function is encapsulated using an object of the class, *CustomGPUObject*. This helps us easily manage data-structures that needs to be worked on both the host-space and the device-space. We then call the class method, *copyFromHostToDevice()*. This allocates space for the output in the GPU global-memory. 
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
Once the inputs have been made available in the GPU global-memory and space has been allocated for the outputs there, we prepare the launch-configuration parameters. Since this particular transformation is rather straightforward, the number of threads in a block is fixed and the number of grids are chosen in such a way that the number of threads along each dimension is greater than or equal to the pixels in each direction. We then launch the kernel into the default stream with the launch-configuration parameters and the function arguments. 
```C
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
```

<!-- ======================================================================= -->
Note that kernel launches to the default stream are blocking. Which means that any lines after this function call is run only after the stream has finished running. Thus, we now copy the outputs from the GPU global-memory back into the host-side using the class method, *copyFromDeviceToHost()*. 
```C
   // Getting data back from device global-memory to host-memory
    outputImage.copyFromDeviceToHost();

    // shutting
    cudaDeviceSynchronize();
    cudaDeviceReset();
```


### Final CUDA Code
Putting it all together, we get the following for the CUDA code. 

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



## References
- MathWorks. “rgb2ycbcr.” MATLAB Documentation, MathWorks, [Link](www.mathworks.com/help/images/ref/rgb2ycbcr.html). Accessed 26 Sept. 2024.
- MathWorks. “ycbcr2rgb.” MATLAB Documentation, MathWorks, [Link](www.mathworks.com/help/images/ref/ycbcr2rgb.html). Accessed 26 Sept. 2024.
- MathWorks. “mxCreateNumericArray.” MATLAB API Reference, MathWorks, [Link](www.mathworks.com/help/matlab/apiref/mxcreatenumericarray.html?searchHighlight=mxcreatenumericarray&s_tid=srchtitle_support_results_1_mxcreatenumericarray). Accessed 26 Sept. 2024.





















