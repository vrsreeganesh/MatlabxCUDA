# Introduction to Constant Memory (not complete)

## Overview
In this chapter, we introduce constant-memory and how to use it through a very simple example. *Constant Memory* is another class of memory that is available in most NVIDIA GPUs. Using constant-memory to store data that is highly fetched, won't be edited and not-too-large for cache results in highly improved performance due to the reduced data-fetch times from the SMs. This chapter presents how to use constant-memory generally and demonstrate how to use constant-memory through an example. 

## Background

### Stencil Operation (complete but not polished)
A stencil operation is a computational technique used in multiple fields like image and signal processing. It involves applying a function or operation to a neighborhood of data points around a central point in a grid or an array. The output of this operation produces values at each point which is a function of  the elements in its neighborhood. 

The following are the key-aspects of a stencil operation 
- **Neighborhood**: The stencil defines a pattern or shape around each point in the grid that forms the neighborhood of that point for this operation. The elements of this neighborhood are used to compute the results that will be stored in the output index corresponding to the input center index. Common stencils for images include 3x3, 5x5 or more complex shapes. Note that the dimensions are usually odd-dimensions along each arrays so that there is a "center element". 
- **Computation**: At each grid point, the operation combines the values of the neighboring points according to the stencil's rules. Basic stencil operations are averaging, summing, max-ing, or more complex functions. 
- **Applications**: Following are three different fields where this is used
    - **Numerical Methods**: In solving partial differential equations (PDEs), stencil operations are used to approximate derivatives and other operations.
    - **Image Processing**: Stencils are applied for tasks like edge detection, blurring, and sharpening, where each pixel's new value depends on the values of its neighboring pixels.
    - **Scientific Computing**: Stencil operations are common in simulations, such as fluid dynamics, where they model the interaction between points in a grid.
Stencil operations are highly parallelizable, making them well-suited for implementation on GPUs using CUDA.

### Constant Memory (complete but  not polished)
*Constant Memory* is another class of memory that is available in most NVIDIA GPUs. The constant memory is at the same distance from the SM as that of the Global Memory. Though constant memory is not at the same proximity as that of shared-memory, it is aggressively cached. This means that using constant memory to store data that will remain constant throughout the process in addition to being regularly used will produce gains in speed. Thus this kind of data that is stored in the constant memory is efficiently and effectively cached. This kind of aggressive caching allows for extreme streamlining and large gains in time due to the high hit rates in the cache. 

Regarding access-privilege, the data in Constant Memory is only allowed to be set from the host-side and never from the device-side. That is, host side write access while device-side has read access. Due to these technical designs and characteristics, Constant Memory must be used to store those values that are 
- Regularly Used
- No edits
- Not too large

Thus, for tasks such as convolution where we use the same kernel throughout the data, using constant-memory to store the kernel-weights is a good idea. Even though most filters will fit just fine into constant-memory of constant sizes, it is important to remember that the improvement in performance stems from the high-hit rate of increasing the probability of finding them in the cache associated with the constant-memory. 

#### Using Constant-Memory (complete but  not polished)
Unlike Shared Memory, *Constant Memory* doesn't give us the option to be dynamically allocated. Thus, we allocate constant memory during compile time, through preprocessor directives. Constant memory is declared in the same way we declare a global variable, but with the addition that we use the identifier, $\_\_constant\_\_$. Once declared, we populate constant-memory using the function, *cudaMemcpyToSymbol()*. Following is a skeletal code that demonstrates using constant memory. 
