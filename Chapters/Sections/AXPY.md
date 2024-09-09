# AXPY Implementation

## C Code

```C
/*
=================================================================================
Aim:
    Implementing AXPY with C
=================================================================================
*/

// headers
#include "mex.h"

// implementing function
void axpy_implementation(double alpha, double *X, double *Y, double *Output, size_t M, size_t N)
{
    // initializing indexing variables
    size_t i, j; 

    // going through each value and producing the transformation
    for(i = 0; i<M; ++i)
    {
        for(j = 0; j<N; ++j)
        {
            Output[j+i*N] = alpha * X[j+i*N] + Y[j+i*N];
        }
    }
}

// gate-way function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // check the number of inputs
    if(nrhs!=3)
    {
        mexErrMsgTxt("Number of Inputs are wrong \n");
    }
    
    // check the number of outputs
    if(nlhs!=1){
        mexErrMsgTxt("Number of outputs are wrong \n");
    }
    
    // check the input data-type
    if(!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])||mxGetNumberOfElements(prhs[0])!=1)
    {
        mexErrMsgTxt("The first argument should be a scalar \n");
    }
    if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]))
    {
        mexErrMsgTxt("The second input must be a matrix of type double \n");
    }
    if(!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]))
    {
        mexErrMsgTxt("The second input must be a matrix of type double \n");
    }
    
    // getting the dimensions of the input
    size_t M0 = mxGetM(prhs[1]);
    size_t N0 = mxGetN(prhs[1]);
    size_t M1 = mxGetM(prhs[2]);
    size_t N1 = mxGetN(prhs[2]);
    
    // check if the dimensions are valid for this operation
    if(M0!=M1 || N0!=N1)
    {
        mexErrMsgTxt("X and Y must have same dimensions \n");
    }
    
    // obtaining pointers to the inputs
    double *inputpointer_A;
    double *inputpointer_B;
    double alpha = mxGetScalar(prhs[0]);
    
#if MX_HAS_INTERLEAVED_COMPLEX
    inputpointer_A = mxGetDoubles(prhs[1]);
    inputpointer_B = mxGetDoubles(prhs[2]);
#else
    inputpointer_A = mxGetPr(prhs[1]);
    inputpointer_B = mxGetPr(prhs[2]);
#endif
    
    // building output
    plhs[0] = mxCreateNumericMatrix(M0, N0, mxDOUBLE_CLASS, mxREAL);;
    
    // getting pointer to output
    double *outputpointer;
#if MX_HAS_INTERLEAVED_COMPLEX
    outputpointer = mxGetDoubles(plhs[0]);
#else
    outputpointer = mxGetPr(plhs[0]);
#endif
    
    // calling function
    axpy_implementation(alpha, inputpointer_A, inputpointer_B, outputpointer, M0, N0);

    // returning
    return;
}
```


## Matlab Code

```Matlab
%{
=================================================================================
Aim:
    Compiling and Running AXPY implementation
=================================================================================
%}

%% Basic Setup
clc; clear; close all;

%% Compiling
mex axpy_implementation.c;

%% Building arguments
X = 1*ones(10)
Y = 8*ones(10)
alpha = 2;

%% Calling mex-function
outputMatrix = axpy_implementation(alpha, X, Y)
confirmMatrix = alpha*X + Y
```
