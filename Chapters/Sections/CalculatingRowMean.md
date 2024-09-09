
# Calculating Row Mean
A very simple example. This mex-function takes in a 2D matrix and returns the average of each row. 
Note that the data is stored in C as a row-major. This means that two successive elements in the same column are separated by the number of columns. 

## C Code

```C
/*
=================================================================================
Aim: Calculating row-means
=================================================================================
*/ 

// headers
#include "mex.h"

// gateway function
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    // Ensuring number of inputs
    if(nrhs!=1)
    {
        printf("Number of inputs are wrong \n");
        return;
    }
    
    // Ensuring number of outputs
    if(nlhs!=1)
    {
        printf("Number of outputs are wrong \n");
        return;
    }

    // Getting pointer to input array
    double *inputpointer = (double *)mxGetPr(prhs[0]);
    int nrows = (int)mxGetM(prhs[0]);
    int ncols = (int)mxGetN(prhs[0]);

    // setting up output
    plhs[0] = mxCreateNumericMatrix(nrows, 1, mxDOUBLE_CLASS, mxREAL);
    double *outputpointer = (double *)mxGetPr(plhs[0]);

    // calculating the average value for each row
    double sum = 0;
    for(int i_row = 0; i_row<nrows; ++i_row)
    {
        // initializing the sum variable
        sum = 0; 

        // Calculating sum of each row
        for(int jcol = 0; jcol<ncols; ++jcol)
        {
            sum += inputpointer[i_row*ncols + jcol];
        }
        
        // finding the average of the current row
        outputpointer[i_row] = (double)(sum/ncols);
    }


    // Returning
    return;
}
```


## MATLAB Code
```Matlab
%{
========================================
Aim:
    Calculating the average of each rows
========================================
%}

%% Basic setup
clc; clear all; close all;

%% Compiling mex-function
mex row_average.c

%% Setting up input arguments
inputmatrix = 2*eye(2);

%% Running the mex-function
output00 = row_average(inputmatrix);
```
