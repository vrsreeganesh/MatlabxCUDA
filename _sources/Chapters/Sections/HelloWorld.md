# Hello World

### Overview
<!-- In this trivial example, we introduce the reader to printing a simple line from the mex-file. The general print function from mex-function is *mexPrintf*. That being said, regularly *fprintf* works just fine. We start with the simple $fprintf$. The call from Matlab to this function results in the code producing the result, "Hello World".  -->

In this trivial example, we introduce the reader to printing a line from the mex-file. The general print function from mex-function is *mexPrintf*. That being said, please note that *fprintf* also works just fine. So we'll start here. 

The following script results in a, "Hello World" output in the console. 

<!-- =============================== -->
<!-- =============================== -->
<!-- =============================== -->
<!-- =============================== -->
<!-- =============================== -->
### C Code
In the C-script, we start off by including the header-file for the mex-functions. We then define the gateway function and start printing the results. 
```C
// Including header-file
#include "mex.h"

// Defining gateway function
void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray *prhs[])
{
    // printing
    printf("Hello World \n");
}
```

<!-- =============================== -->
<!-- =============================== -->
<!-- =============================== -->
<!-- =============================== -->
<!-- =============================== -->
### MATLAB Code
In the Matlab script, we straight up compile the mex-file using the keyword, "mex". Note that since the script is rather straightforward, there is no need for additional linking. Once the compilation is completed, we call the function. 


```Matlab
%% Aim: Printing hello world

%% Basic setup
clc; clear; close all;

%% Compiling
mex HelloWorld.c

%% Calling mex-function
HelloWorld()
```
