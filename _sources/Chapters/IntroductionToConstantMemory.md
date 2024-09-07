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

