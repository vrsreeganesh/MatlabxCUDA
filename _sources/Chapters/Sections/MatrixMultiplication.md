# Matrix Multiplication
Here, we present a simple code that multiplies two matrices using Mex. The code takes two double-type matrices. Calculates their matrix-multiplied product and returns it.  

## C Code
```C
/*
=================================================================================
    Aim:
        Matrix Multiplication
=================================================================================
*/ 

// header
#include "mex.h"

// matrix-multiplication function
void matmul(double *A, double *B, double *C, size_t M0, size_t N0, size_t N1)
{
    // the two values
    mwSize i, j, k; 
    double prodsum = 0;

    // calculating the matrix multiplication
    for(k = 0; k<N1; ++k)
    {
        for(i = 0; i<M0; ++i)
        {
            // finding the inner product
            prodsum = 0;
            for(j = 0; j<N0; ++j)
            {
                prodsum += A[i*N0 + j] * B[j*N1 + k];
            }

            // assigning values 
            C[i*N0 + k] = prodsum;
        }
    }
}

// gate-way function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // check the number of inputs
    if(nrhs!=2)
    {
        mexErrMsgTxt("Number of Inputs are wrong \n");
    }
    
    // check the nummber of outputs
    if(nlhs!=1)
    {
        mexErrMsgTxt("Number of outputs are wrong \n");
    }
    
    // check the data type of inputss
    if(!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]))
    {
        mexErrMsgTxt("Input data-type is wrong \n");
    }
    
    // getting dimensions from inputs
    size_t M0 = mxGetM(prhs[0]);
    size_t N0 = mxGetN(prhs[0]);
    size_t M1 = mxGetM(prhs[1]);
    size_t N1 = mxGetN(prhs[1]);
    
    // check if dimensionality is compatible for operation
    if(N0!=M1)
    {
        mexErrMsgTxt("Input dimensions are not compatibility for matrix multiplication \n");
    }
    
    // getting pointers to inputs
    double *inputpointer_A; 
    double *inputpointer_B;
#if MX_HAS_INTERLEAVED_DATA
    inputpointer_A = mxGetDoubles(prhs[0]);
    inputpointer_B = mxGetDoubles(prhs[1]);
#else
    inputpointer_A = mxGetPr(prhs[0]);
    inputpointer_B = mxGetPr(prhs[1]);
#endif
    
    // setting up output based on information
    plhs[0] = mxCreateNumericMatrix(M0, N1, mxDOUBLE_CLASS, mxREAL);
    
    // getting pointer to output
    double *outputpointer;
#if MX_HAS_INTERLEAVED_DATA
    outputpointer = mxGetDoubles(plhs[0]);
#else
    outputpointer = mxGetPr(plhs[0]);
#endif
    
    // calling function
    matmul(inputpointer_A, inputpointer_B, outputpointer, M0, N0, N1);

    // return
    return;
}
```


## MATLAB Code
```Matlab
%{
=================================================================================
    Aim:
        Matrix multiplying two 2D Arrays
=================================================================================
%}

%% Basic Setup
clc; clear; close all;

%% Compiling Code
mex matrixmultiply.c

%% Setting up input 
matrixA = ones(10)
matrixB = 0.1*ones(10)

%% Calling the function
outputmatrix = matrixmultiply(matrixA, matrixB)
```
