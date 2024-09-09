# Element-wise multiplying two 2D-matrices
Here, we see a simple example of element-wise multiplying two matrices using mex. No CUDA stuff. Not yet. 



## C Code

```C
/*
=================================================================================
Aim:
    Element-wise multiply two 2D matrices 
=================================================================================
*/ 

// header files
#include "mex.h"

// the function
void elementwisemultiply(double *inputpointer_0, double *inputpointer_1, double *outputpointer, int numelements)
{
    for(size_t i = 0; i<numelements; ++i)
    {
        outputpointer[i] = inputpointer_0[i]*inputpointer_1[i];
    }
}

// gate-way function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // check the number of inputs
    if(nrhs!=2)
    {
        mexErrMsgTxt("The number of inputs are wrong \n");
    }

    // check the number of outputs
    if(nlhs!=1)
    {
        mexErrMsgTxt("The number of outputs are wrong \n");
    }

    // check the input data-type
    if(!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]))
    {
        mexErrMsgTxt("The inputs are expected to be of type, double \n");
    }

    // check the dimension compatibility
    size_t M0 = mxGetM(prhs[0]);
    size_t N0 = mxGetN(prhs[0]);
    size_t M1 = mxGetM(prhs[1]);
    size_t N1 = mxGetM(prhs[1]);


    // setting up pointers to the inputs
    double *inputpointer_0;
    double *inputpointer_1;

    // extracting pointers to the inputs
#if MX_HAS_INTERLEAVED_COMPLEX
    inputpointer_0 = mxGetDoubles(prhs[0]);
    inputpointer_1 = mxGetDoubles(prhs[1]);
#else
    inputpointer_0 = mxGetPr(prhs[0]);
    inputpointer_1 = mxGetPr(prhs[1]);
#endif

    // setup outputs 
    plhs[0] = mxCreateNumericMatrix(M0, N0, mxDOUBLE_CLASS, mxREAL);

    // setup pointer to output
    double *outputpointer;
#if MX_HAS_INTERLEAVED_COMPLEX
    outputpointer = mxGetDoubles(plhs[0]);
#else
    outputpointer = mxGetPr(plhs[0]);
#endif

    // call the function
    elementwisemultiply(inputpointer_0, inputpointer_1, outputpointer, (size_t)M0*N0);

    // returning
    return;
   
}
```


## Matlab Code

```Matlab
%{
=================================================================================
    Aim:
        Element-wise multiplying two matrices
=================================================================================
%}

%% Basic setup
clc; clear; close all;

%% Compiling mex-function
mex elementwisemultiply.c

%% Setting up input arguments
inputmatrix_A = 2*ones(10);
inputmatrix_B = 5*ones(10);

%% Calling the function
outputmatrix = elementwisemultiply(inputmatrix_A, inputmatrix_B);
```