# Calculating Row Mean

## Overview
Here, we visit a simple example of calculating the row-mean of an input matrix. The mex-function takes in a 2D matrix and returns the average of each row. 

<!-- A very simple example. This mex-function takes in a 2D matrix and returns the average of each row. 
Note that the data is stored in C as a row-major. This means that two successive elements in the same column are separated by the number of columns.  -->

<!-- =============================== -->
<!-- =============================== -->
<!-- =============================== -->
<!-- =============================== -->
<!-- =============================== -->
## C Code
In the following script, we start off by including the header-file and define the gateway function. 

In the gateway function, we start by checking the arguments. We first check the number of inputs passed from the Matlab script. This is followed by the number of outputs expected by the Matlab script. Once the tests have been passed, we assign a pointer to hold the address of the input, we obtained using the function, *mxGetPr()*. We then calculate the number of rows and columns. 

Once the inputs have been maintained, we allocate space for the output using [*mxCreateNumericMatrix()*](https://www.mathworks.com/help/matlab/apiref/mxcreatenumericmatrix.html). The arguments to the function are 
- number of rows
- number of columns
- data-type of matrix to be created
- real or imaginary 

The function creates a matrix described by the arguments and returns a pointer. We've allocated the contents of the pointer to *plhs[0]*, which contains the first output. The pointer to this is then stored into the variable, *outputpointer*. 

```C
/* =======================
Aim: Calculating row-means
======================= */ 

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
    plhs[0] = mxCreateNumericMatrix(nrows, 
                                    1, 
                                    mxDOUBLE_CLASS, 
                                    mxREAL);
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
