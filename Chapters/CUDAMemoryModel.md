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


## Shared Memory
Shared Memory is another kind of memory that is available on most NVIDIA-GPUs. Unlike Local Memory and Global memory which reside on-device and far from the SMs, shared memory reside on-chip. Due to the physical proximity in addition to the high-bandwidth channels, the latency for read-and-writes are extremely low and magnitudes lower than that of Global and Local Memory. It is designed to support efficient, high-bandwidth sharing of data among threads in a block. 

Regarding access, the contents of the shared memory are available to all threads within a block. This means that data-structures allocated to shared memory is available to all threads within a block. And each block gets their copy of the structure that was assigned to shared memory. For example, if one were to allocate a double array of length 32, all the blocks will have their own copy of this array. This kind of read-and-write access allows the threads within a block to collaborate to carry out different tasks. 

Shared Memory can be assigned in two different ways: statically and dynamically. Static allocation allows for higher-dimensional arrays. However, dynamic allocation only allows for linearized arrays. To use the linear array to represent higher-dimensional arrays, we'll need to produce work arounds. 

## Registers
Registers are another kind of memory that is available on most NVIDIA-GPUs. Like Shared Memory, this is on-chip. However, registers are a lot closer to the ALU than Shared Memory and connected to it through high-bandwidth connections. This high-speed connection combined with close proximity allows read-and-write operations to have extremely low-latencies. 

Regarding privilege, the contents of a thread's registers are exclusively only for that register. Even threads within the same block do not have access to other thread's registers, except for some advanced conditions known as warp-shuffles. Primarily, registers are used by kernel functions to hold frequently accessed variables that private to each thread. In addition, data that is fetched from other memories are stored here in order to perform operations on it. 