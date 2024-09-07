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

