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

## CUDA

### Kernel
The kernel takes in the following set of arguments
1. Pointer to memory holding input-data
2. Sampling factor
3. Pointer to memory allocated for output-data
4. Pointer to array holding input-dimensions
5. Number of output-dimensions  

Thus the first part of the kernel definition looks is the following 
```C
// global function
__global__ void upsampling_ip_cuda(double *d_inputPointerA,
                                   const int samplingFactor,
                                   double *d_outputPointer,
                                   const mwSize *d_inputDimensionsA,
                                   mwSize *d_outputDimensions)
{
    ...
}
```

The mapping from thread-address to the input-addressing is straightforward. Each dimension of the thread is allocated to each pixel. After mapping, we obtain the linear mapping from this coordinate to the linear coordinates. 
```C++
    // getting thread coordinates
    const int tidx = threadIdx.x + blockIdx.x*blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y*blockDim.y;
    const int tidz = threadIdx.z + blockIdx.z*blockDim.z;

    // getting linear index into the input
    const int linearIndex2D_Input = \
                tidx + \
                tidy*d_inputDimensionsA[0] + \
                tidz*d_inputDimensionsA[0]*d_inputDimensionsA[1];
```


We then calculate the coordinates of the destination matrix. We calculate the relevant dimensions with the sampling factor since its upsampling. Refer the above equation if this step doesn't make sense. And just like before, we  calculate the linear index into the output-memory. 
```C
   // finding the destination coordinates
    const int tidx_output = tidx*samplingFactor;
    const int tidy_output = tidy*samplingFactor;
    const int tidz_output = tidz;
```

Due to the logic of the upsampling, we copy the values from each index of the input to the corresponding scaled index in the output-memory in the following manner. 

```C
   // copying the values to the output
    if(tidx_output<d_outputDimensions[0] && tidy_output<d_outputDimensions[1])
    {
        d_outputPointer[linearIndex2D_Output] = \
            (double)d_inputPointerA[linearIndex2D_Input];
    }
```

This is followed by syncing the threads. The final kernel definition must look like this
```C
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

```




### Main

In the main function, we do the following set of steps  

1. Receive and check input-arguments
2. Setup space for outputs
3. Call CUDA function
4. Prepare output for Matlab


We first start with checking the validity of inputs. The first check is the number of inputs and the number of expected outputs. Then we check if both the inputs are of type double. Note that even though the second argument is expected to be an integer, we check if its type, *double*, since Matlab, by default, stores everything as a *double*. 

```C
void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray *prhs[])
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
    
    ...
}
```


Next, we receive the two inputs. The image is received using our custom class, *CustomGPUObject*. And the sampling factor is received using a simple scalar. (Note that the reason we're not using the class object for this because scalars can be transferred to the GPU memory-space without the CUDA API functions). Once the inputs have been received and stored to their appropriate data-structure, we send the input-image to the GPU memory-space using the class method, *copyFromHostToDevice()* . 


```C
    // setting up inputs
    CustomGPUObject inputImage(prhs[0]);
    const int samplingFactor = (int)mxGetScalar(prhs[1]);
    inputImage.copyFromHostToDevice();
 ```

Next, we setup the output-dimensions. We first allocate space in the CPU for the output-dimensions. Then we copy the expected output-dimensions to this array. 
```C
    // making output's dimension-array
    mwSize *outputDimensions = (mwSize *)malloc(inputImage.numDimensions*sizeof(mwSize));
    memcpy(outputDimensions, inputImage.inputDimensions, inputImage.numDimensions*sizeof(mwSize));
    outputDimensions[0] = inputImage.inputDimensions[0]*(mwSize)samplingFactor;
    outputDimensions[1] = inputImage.inputDimensions[1]*(mwSize)samplingFactor;
```

<!-- ======================================================================= -->
The array that holds the output-dimensions are then used to create a Matlab matrix using the function, [*mxCreateNumericArray*](https://www.mathworks.com/help/matlab/apiref/mxcreatenumericarray.html?searchHighlight=mxcreatenumericarray&s_tid=srchtitle_support_results_1_mxcreatenumericarray). This created matrix is then used to create an object of our class, *CustomGPUObject*. We use this object instead of the pointer directly to make coding easier. As usual, since our work is on the GPU, we use the class method, *copyFromHostToDevice* to make memory for the output and sending the output-dimensions to the GPU memory. 

```C
   // Setting up output
    plhs[0] = mxCreateNumericArray(inputImage.numDimensions,
                                   outputDimensions,
                                   mxDOUBLE_CLASS,
                                   mxREAL);
    CustomGPUObject outputImage(plhs[0]);
    outputImage.copyFromHostToDevice();
 ```


<!-- ======================================================================= -->
We then prepare for the kernel-launch. We stick to a block dimension of $32 \times 32$. The number of blocks are chosen such that there is sufficient number of blocks available to make sure all the pixels are processed. 
```C
    // preparing for function call
    dim3 blockConfiguration(32,32,1);
    dim3 gridConfiguration((int)(inputImage.inputDimensions[0]+blockConfiguration.x-1)/blockConfiguration.x,
                           (int)(inputImage.inputDimensions[1]+blockConfiguration.y-1)/blockConfiguration.y,
                            max(1, (int)inputImage.inputDimensions[2]));
```

<!-- ======================================================================= -->
We then launch the kernel with the appropriate arguments and launch-configuration parameters. 
```C
    // calling function
    upsampling_ip_cuda<<<gridConfiguration,
                         blockConfiguration>>>(inputImage.d_inputPointer_real,
                                               samplingFactor,
                                               outputImage.d_inputPointer_real,
                                               inputImage.d_inputDimensions,
                                               outputImage.d_inputDimensions);
```
<!-- ======================================================================= -->
Kernel launches to the default stream is blocking. Which means that the rest of the code is run only after the function has finished its processing. So the next line involves copying the results back to make it available in the host-side. This is followed by some maintenance functions. 
```C
// fetching data from device global-memory to host memory
    outputImage.copyFromDeviceToHost();

    // shutting down
    cudaDeviceSynchronize();
    cudaDeviceReset();
 ```

### Final CUDA Code
Putting together the CUDA code, we get the following
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


## Reference
- MathWorks. “Phantom.” MATLAB Documentation, MathWorks, [Link](www.mathworks.com/help/images/ref/phantom.html#d126e261153). Accessed 26 Sept. 2024.
- MathWorks. “mxCreateNumericArray.” MATLAB API Reference, MathWorks, [Link](www.mathworks.com/help/matlab/apiref/mxcreatenumericarray.html?searchHighlight=mxcreatenumericarray&s_tid=srchtitle_support_results_1_mxcreatenumericarray). Accessed 26 Sept. 2024.