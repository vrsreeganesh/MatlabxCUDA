# Introduction to Shared Memory


## Overview
In this chapter, we introduce shared-memory and how to use it through a very simple example. Shared-memory is a class of on-chip memory in NVIDIA GPUs that is primarily used by threads, within a thread-block, to collaborate. Convolution is a mathematical operation on two functions that produces a third function. The process of convolution calls for the threads to collaborate, due to the accumulative nature of the operation. Thus, making it an excellent example to present the use of shared-memory. 

## Background 

### Convolution Theory
<!--% What is convolution??   -->
Convolution is a mathematical operation on two functions that produces a third function. It is defined as the integral of the product of the two functions after one is reflected about the y-axis and shifted. The integral is evaluated for all values of shift, producing the convolution function.

$$
(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g (t - \tau) d\tau
$$

<!-- why is convolution important?? -->
The process of convolution is important because it serves as a fundamental operation in many fields, including signal processing, image processing, and neural networks. Convolution allows for the systematic combination of two functions or datasets, capturing how the shape of one is modified by the other. In practical terms:

- **Signal Processing**: Convolution helps filter signals by enhancing or suppressing specific features, such as removing noise or highlighting certain frequencies.

- **Image Processing**: In images, convolution is used to apply filters, such as blurring, sharpening, edge detection, and more. This is done by sliding a filter (kernel) over the image to create a new transformed image.

- **Neural Networks**: In convolutional neural networks (CNNs), convolution operations are used to automatically learn and extract features from raw data, like edges, textures, or patterns in images, which are crucial for tasks like image classification or object detection.

In essence, convolution is a versatile tool for transforming data in ways that reveal or emphasize important characteristics, making it invaluable in both analysis and practical applications.



### Shared Memory
Shared Memory is another kind of memory that is available on most NVIDIA-GPUs. Unlike Local Memory and Global memory which reside on-device and far from the SMs, shared memory resides on-chip. Due to the physical proximity in addition to the high-bandwidth channels, the latency for read-and-writes are extremely low and magnitudes lower than that of Global and Local Memory. It is designed to support efficient, high-bandwidth sharing of data among threads in a block. 

Regarding access, the contents of the shared memory are available to all threads within a block. This means that data-structures allocated to shared memory is available to all threads within a block. And each block gets their copy of the structure that was assigned to shared memory. For example, if one were to allocate a double array of length 32, all the blocks will have their own copy of this array. This kind of read-and-write access allows the threads within a block to collaborate to carry out different tasks. 

Shared memory can be assigned in two different ways: static and dynamic. Static allocation allows for higher-dimensional arrays with the trade-off being that the dimensions should be known before-hand. Dynamic allocation, on the other hand, although only allowing for linear arrays, it allows for run-time allocation of shared-memory based on some variable value. To have dynamic high-dimensional arrays, one will have to bring in custom classes that explicitly take care of the expected functionality. The following sections show the two different skeletal ways of allocating shared-memory: static and dynamic. 


#### Shared Memory Allocation: Static
In static allocation, we decide on the size of the available shared memory at compile-time. The standard method of assigning this is through pre-processor directives. Static allocation allows the shared-memory to have higher dimensions. The following shows the creation of a 2-Dimensional shared matrix with the help of preprocessor directives. Note that in static allocation of shared-memory, one does not need to pass the size of the shared-memory as the third kernel-launch parameter. It will be automatically inferred through the preprocessor directives and data-type. 

```C++
// Headerfiles
#include ``mex.h''


// shared-memory related parameters
#define SHAREDMEMX 7
#define SHAREDMEMY 7

// kernel
__global__ mockkernel()
{
	__shared__ sharedMatrix[SHAREDMEMX][SHAREDMEMY];
	...
}

// gate-way function
void mexFunction(...)
{
	...
	// Initialize thread parameters
	dim3 blockspergrid;
	dim3 threadsperblock;
	
	// shared memory related parameters
	int sharedmemoryLength = 128;
	
	// calling the kernel
	kernelfunction<<<blockspergrid, threadsperblock>>>();
	
	
}
```




