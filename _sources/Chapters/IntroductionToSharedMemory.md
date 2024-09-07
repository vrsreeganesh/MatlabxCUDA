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

