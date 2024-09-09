# Multiplying a row-matrix with a scalar

Here, we see how we can multiply a simple row-matrix with a scalar

## C Code

```C
/*
=================================================================================
Aim:
    Multiplying each element of an array with a product. 
=================================================================================
*/ 

// headers
#include "mex.h"

// the function
void arrayProduct(double *inputarray, double multiscalar, double *outputarray, mwSize numelements)
{
    // multiplying each element with the scalar
    for(mwSize i = 0; i<numelements; ++i)
    {
        outputarray[i] = multiscalar * inputarray[i];
    }
}

// gateway function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // checking number of inputs
    if(nrhs!=2)
    {
        printf("Number of Inputs are Wrong \n");
    }
    
    // checking number of outputs
    if(nlhs!=1)
    {
        printf("Number of Outputs are Wrong \n");
    }

    // Checking if the first input is an array and of type, double
    if(!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
    {
        printf("Input array was expected to be of type, double \n");
        return;
    }

    // Checking if second input is a scalar of type, double
    if(!mxIsDouble(prhs[1])  || mxGetNumberOfElements(prhs[1])!=1)
    {
        printf("The second input is of wrong type or size \n");
        return;
    }

    // ensuring that the input is a row-vector
    if(mxGetM(prhs[0])!=1)
    {
        printf("The input should be a row vector \n");
        return;
    }

    // extracting the value of the input scalar
    double multiplier = mxGetScalar(prhs[1]);

    // extracting a pointer to the data in the input argumetns
    double *inputpointer;
#if MX_HAS_INTERLEAVED_COMPLEX
    inputpointer = mxGetDoubles(prhs[0]);
#else
    inputpointer = mxGetPr(prhs[0]);
#endif

    // getting dimensions of input matrix
    size_t ncols = (size_t)mxGetN(prhs[0]);

    // Creating the output matrix
    plhs[0] = mxCreateNumericMatrix(1, (mwSize)ncols, mxDOUBLE_CLASS, mxREAL);

    // get a pointer to the real data in the output matrix
#if MX_HAS_INTERLEAVED_COMPLEX
    double *outputpointer = mxGetDoubles(plhs[0]);
#else
    double *outputpointer = mxGetPr(plhs[0]);
#endif

    // calling the computational routine
    arrayProduct(inputpointer, multiplier, outputpointer, (mwSize)ncols);

    // return control
    return;
}
```

## MATLAB Code

```Matlab
%{
=================================================================================
    Aim: Mexcuda for multiplying a row array with a constant
=================================================================================
%}

%% Basic setup
clc; clear; close all;

%% Compiling mex-function
mex arrayProduct.c;

%% Building arguments
inputarray = [1,2,3,4,5];
multiplier = 10;

%% calling the function
outputarray = arrayProduct(inputarray, multiplier)
```