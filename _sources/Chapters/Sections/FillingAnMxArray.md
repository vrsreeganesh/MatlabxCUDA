# Filling an mxArray

This is a simple example of filling an mxArray. The C code takes no arguments. It contains a pre-defined array. We create a numeric matrix and then copy those values onto it. A rather simple code but its always important to start things slow. 

Use the mxSetDoubles and mxGetDoubles functions for data of type double. For numeric data other than double, use the one of the typed data access functions.

## C Code

```C
/*
=================================================================================
Aim: 
    A simple programming illustrating the filling of an mxArray
Notes:
=================================================================================
*/
#include "mex.h"

/* The mxArray in this example is 2x2 */
#define ROWS 2
#define COLUMNS 2
#define ELEMENTS 4

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // checking number of inputs
    if(nlhs!=1)
    {
        printf("Number of inputs are wrong \n");
        return;
    }

    // checking number of outputs
    if(nrhs!=0)
    {
        printf("Number of outputs are wrong \n");
        return;
    }

    // Setting up data locally 
#if MX_HAS_INTERLEAVED_COMPLEX
    mxDouble* outputpointer;
    const mxDouble data[] = {1.0, 2.0, 3.0, 4.0};
#else
    double *outputpointer;
    const double data[] = {1.0, 2.0, 3.0, 4.0};
#endif

    // Setting up output
    plhs[0] = mxCreateNumericMatrix(ROWS, COLUMNS, mxDOUBLE_CLASS, mxREAL);

    // getting the pointer to the output object
#if MX_HAS_INTERLEAVED_COMPLEX
    outputpointer = mxGetDoubles(plhs[0]);
#else
    outputpointer = mxGetPr(plhs[0]);
#endif

    // Copying the data to the output object
    for(mwSize i = 0; i<ELEMENTS; ++i)
    {
        outputpointer[i] = data[i];
    }

    // returning 
    return;
}
```


## MATLAB Code

```Matlab
%{
=================================================================================
    Aim:
        Calling a function that fills a mxArray
=================================================================================
%}

%% Basic setup
clc; clear all; close all;

%% Compiling mex-function
mex arrayFillGetPr.c;

%% Calling mex-function
outputmatrix = arrayFillGetPr()
```