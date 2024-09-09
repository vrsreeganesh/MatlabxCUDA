# CUDA Memory Model

## Overview
NVIDIA GPUs incorporate a variety of memory types. They differ in their size, latency, throughput, permissions and other characteristics. Having an understanding of the different kinds of memories allows us to use them  accordingly such that code-performance is maximised. The memories introduced and discussed in this chapter are by no means, exhaustive. But it does contain the set of memories that a beginner or an intermediate must learn how to use in order to write effective CUDA code. Readers are highly encouraged to refer to the official CUDA Programming Guide for a deeper learning and understanding of NVIDIA-GPU memories. 

## Global Memory 
Global Memory is a type of memory in NVIDIA-GPUs that, as the name suggests, can be accessed by all the threads in the grid. It is usually the largest, with the modern day Global Memories being GBs in size. They are also the slowest. However, we usually send the data that we want to process using kernels to the Global Memory first owing to its size. Due to the presence of our data in the global memory and its high-latency characteristic, reducing the amount of access to the global memory is a rule of thumb that must be followed by cuda programmers. More on this later. 

The basic protocol when writing CUDA code is to send the data that you want to process to the global memory first and then have the threads fetch it according to the sections they're responsible for, produce the results and write it somewhere in the global memory. The following section presents how to transfer data between host and device global-memory. 


## Local Memory
Local Memory is another on-device memory that is available on NVIDIA-GPUs. The memory allows for both read and write processes from device-side. Despite the name, Local Memory is at a close proximity to the Global Memory. Thus, it too has the high-latency issues. The, ``Local'', part of the name stems from its privilege. The data stored in this memory can only be accessed by the threads to which it belongs to. 

Local Memory is used to store data that cannot be stored in Registers. The kind of data that is usually stored in Local Memory are statically allocated arrays, spilled registers and other elements of the thread's call stack. 

Unlike Global Memory, there is no need for explicit declaration and initialization for using Local Memory. This takes place automatically. This is important because in time-sensitive contexts, it is important to not use data-structures that warrants it storage in Local Memory as fetching or writing to it will incur the same time-delay as that of using Global Memory. This understanding of when and what variables will be stored in Local Memory will help you design optimal kernels. 