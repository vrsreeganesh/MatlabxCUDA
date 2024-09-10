# The Skeletal Structure
In this section, we introduce the mex-cuda structure that we will be following throughout the examples. Note that there are a number of ways to interface CUDA with Matlab using Mex. However, we start with the simple approaches and build up the more difficult methods as we go through. 


## Gateway Function
The first is the gateway function. The gateway function in Mex is not-unlike the main function. It is the function that is called by matlab when we call a Mex code. It takes in mainly four arguments. 

- nlhs: the number of outputs we expect the function to return
- *plhs[]: an array of pointers that is expected to contain the pointers to the structures that we're returning. So if we're returning two arrays as results, then the first element will contain the pointer to the first array and the second element will contain the pointer to the second array. 
- nrhs: the number of inputs that were passed during this function call
- *prhs[]: an array of containing pointers that point to the different data structures that are made available to this particular function. 

The function is defined in the following manner
```C
// gateway function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	...
}
```








## Checking number of Inputs and Outputs


Once the mex gateway function has been defined, the next step is to ensure that the arguments passed to it are the ones that we're expecting. The first set of checks is to ensure that the number of inputs and outputs are correct. CHecking both are rather straightforward. One point of confusion might be that how does one check the number of outputs, ahead of time. So what we mean by nrhs is the number of outputs matlab is expecting rather than seeing ahead the number of outputs this code will produce. In the following segments, we show different examples of when matlab expects different number of outputs

Once the mex gateway function has been defined, the next step is to ensure that the arguments passed to it are valid and expected. The first set of checks is to ensure the number of inputs and expected outputs. (One point of confusion might be how does one check the number of outputs, ahead of time. nrhs refers to the number of outputs expected by the matlab code). The following code segment shows what we mean by the number of outputs expected by matlab. This is automatically populated and passed to the mex gateway function. 

<!-- 
Once the mex gateway function has been defined, the next step is to ensure that the arguments passed to it are the ones that we're expecting. The checks primarily depend on the kind of code and logic that you're defining but the fundamental check
- The number of inputs
- Number of outputs (One point of confusion might be that how does one check the number of outputs, ahead of time. So what we mean by nrhs is the number of outputs matlab is expecting rather than seeing ahead the number of outputs this code will produce) -->

```matlab
// expects one output
outputMatrix = firstFunction();

// expects two outputs
[outputMatrixA, outputMatrixB] = secondFunction();

// expects three outputs
[outputMatrixA, outputMatrixB, outputMatrixC] = thirdFunction();
```

<!-- 
Since nlhs and nrhs are integers, regular C checking should work. We then use a function present in Mex API that allows us to signal to MATLAB that an error has occurred. There are a number of ways to signal to MATLAB that an error has occurred but we choose the simplest. THere are more sophisticated methods to signal to MATLAB about this but we choose to go with the most simple functions, mxErrMsgTxt.  -->

Since *nlhs* and *nrhs* are integers, we check using regular *if* conditions. If the check fails, we use a Mex-API function to signal to MATLAB that an error has occured. Mex-API allows multiple ways to communicate this but here, we present the simplest, *mxErrMsgTxt*. This function indicates to MATLAB that an error has occured and it must display the string argument provided to the function. 

```C
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// checking number of inputs
	if(nrhs!=2)
	{
		mexErrMsgTxt(``Number of Inputs are Wrong \n'');
	}
	
	// checking the number of outputs
	if(nlhs!=1)
	{
		mexErrMsgTxt(``Number of Expected Outputs are Wrong \n'');
	}
}
```


## Checking the Data-types of Inputs

<!-- Once the number of inputs and outputs are checked, we check the data-type of the inputs. This is again done with the help of a set of functions provided by the Mex-API. There are a number of functions that help with this. To following list gives the basic set.  -->

Next, we check the data-type of the inputs. Mex-API provides a number of functions to do the same. Some examples are
- mxIsDouble()
- mxIsComplex()
- mxGetNumberOfElements()
- list five more


Based on the data-structures you're working with, you'll have to mix and match these functions to make sure that the argument meets the conditions of the data-structure that we're expecting. The following shows some basic tests. 

The following shows the basic way of testing whether an incoming data-structure is a double variable, array or matrix. Note that we assume it is the first input argument. The first datatype-based check basically checks if the data is of type double and not complex. 
```C
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// checking number of inputs
	...
	// checking number of expected outputs
	...
	
	// checking the input argument data-type
	if(!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
	{
		mexErrMsgTxt("The Input is expected to be an array of type, double \n";
	}
	
	// Rest of Code
	...
}
```


## Obtaining Input Dimensions

Once we make sure that we've received the right number of inputs, checked the outputs and then tested for the right data-type, we shall now find the way to obtain the dimensions of the input data. The following shows a simple way of obtaining the dimensions of the incoming array/matrix. 

There are a number of functions used to get the dimensions of the input architectures. Some of them are


- mxGetDimensions(): returns array containing the dimensions of the inputs. 
- mxGetNumberOfElements(): returns the number of elements in the input data-structure (an array or a matrix);
- mxGetNumberOfDimensions(): this returns the dimensionality of the input argument. 



```C
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// check number of inputs
	...
	// check number of outputs
	...
	// check input data-type
	...
	
	// obtaining dimensions of the array
	const mwSize *inputDimensions = mxGetDimensions(prhs[0]);
	const mwSize numElements = mxGetNumberOfElements(prhs[0]);
	const mwSize numDimensions = mxGetNumberOfDimensions(prhs[0]);
	
	// Rest of Code
	...
}
```


## Obtaining Pointers to Input Arguments

Now, we obtain the pointers to the input data-structures. The data structures: arrays or matrices are stored in the row-major format adhering to the conventions of C. So we access the input data-structures using a pointer. Using the pointer and the dimension information, we obtain the data values accordingly. There are a number of ways to obtain the pointers to the input data-structures. For now, we stick to the basic way of doing this, using, mxGetPr(). 

```C
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// checking the number of inputs
	...
	// checking the number of outputs
	...
	// checking the input data types
	...
	// obtaining dimensionality of the input arguments
	...
	
	// Obtaining pointer to the input argument
	double *inputPointer;
#if MX_HAS_INTERLEAVED_COMPLEX
	inputPointer = mxGetDoubles(prhs[0]);
#else
	inputPointer = mxGetPr(prhs[0]);
#endif

	// Rest of Code
	...
	
}
```


## Setting Up Output Structure
Once we've created whatever data-structure we have in mind, we create an output. To create a matrix output, there are a number of ways to create this

\begin{itemize}
	\item mxCreateNumericMatrix()
\end{itemize}

The following shows a simple example of assigning an output to the plhs

```C
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// check the number of inputs
	...
	// check the number of outputs
	...
	// check the input data type
	...
	// obtain the pointers to dimensionality
	...
	// obtain pointer to the input data-structures
	...
	// perform operations
	...
	
	// creating output
	plhs[0] = mxCreateNumericMatrix(numRows, numCols, mxDOUBLE_CLASS, mxREAL);
	
	// Rest of Code
	...
}
```



## Obtaining Pointer to Output
Once we've setup the output data structure at plhs, we need to now obtain a pointer to it so that we can start copying the data to the structure. We obtain the pointers in a similar manner to how we obtained the pointer to the input arguments. Note that once the pointer to the output has been obtained, we use this pointer to store the results to. And once the mex-function ends, the data that was copied to this pointer is available at MATLAB as the data-structure that was returned. 

```C
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// checking the number of inputs
	...
	// checking the number of outputs
	...
	// checking the input data type
	...
	// obtaining dimensionality of inputs
	...
	// Obtaining pointers to the inputs
	...
	// Computation/Processing
	...
	// Creating output
	...
	
	// Obtaining pointers to the output
	double *outputPointer; 
#if MX_HAS_INTERLEAVED_COMPLEX
	outputPointer = mxGetDoubles(plhs[0]);
#else
	outputPointer = mxGetDoubles(plhs[0]);
#endif

	// copying data from result to output pointer
	...
	
	// rest of the code
	
}
```


## Miscellaneous
make sure that the data is cleaned-up through free() and cudaFree(). 


