## Hello World

## Overview
In this trivial example, we introduce the reader to printing a simple line from the mex-file. The general print function from mex-function is *mexPrintf*. That being said, regularly *fprintf* works just fine. We start with the simple $fprintf$. The call from Matlab to this function results in the code producing the result, "Hello World". 

<!-- 
============================================================
============================================================
============================================================
 -->
### C Code
```C
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], 
        				 int nrhs, const mxArray *prhs[])
{
    printf("Hello World \n");
}
```

<!-- 
============================================================
============================================================
============================================================
 -->
### MATLAB Code
```Matlab
%% Aim: Printing hello world

%% Basic setup
clc; clear; close all;

%% Compiling
mex HelloWorld.c

%% Calling mex-function
HelloWorld()

```
