## Hello World
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
