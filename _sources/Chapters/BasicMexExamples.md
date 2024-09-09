# Basic Mex Examples

## Overview
This section is designed to get beginners up-to-speed on very simple use cases of Matlab Executable C/C++ code. Knowledge of using mex is important before proceeding to learn mex-cuda as we'll be calling CUDA functions within the same framework. 

## Hello World

% C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
% C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
% C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
% C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
\section{Basic: Hello World}\index{Sectioning!Sections}
Here, we present a very simple example that shows how to print from the mex-function. The mex-function takes no arguments, prints the statement and just return it. 
A very introductory example, yes. MATLAB calls the function and it just prints a, ``Hello World''. 

### C Code
```C
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], 
        				 int nrhs, const mxArray *prhs[])
{
    printf("Hello World \n");
}
```




### MATLAB Code
```Matlab
%{
=================================================================================
Aim: Printing hello world
Notes:
    Compile mex-file containing hello-world.
    running hello-world
=================================================================================
%}

%% Basic setup
clc; clear; close all;

%% Compiling
mex HelloWorld.c
fprintf("----------------------------- \n");

%% Calling mex-function
HelloWorld()

```
