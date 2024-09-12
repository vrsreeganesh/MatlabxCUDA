# Miscellaneous


## Class CustomGPUObject
The following object is used for the CUDA code. 

```C++
// Custom Class
class CustomGPUObject
{
public:

    // Properties::the Host Parts
    double *inputPointer_real;
    double *inputPointer_imag;
    mwSize *inputDimensions; 
    mwSize numDimensions;
    mwSize numElements;

    // Properties::the device parts
    double *d_inputPointer_real;
    double *d_inputPointer_imag;
    mwSize *d_inputDimensions;

    // Properties::meta-data
    int cuda_space_allocation_flag;

    // destructor
    ~CustomGPUObject()
    {
        // freeing the data from the device
        // cudaFree(d_inputPointer_real);
        // cudaFree(d_inputPointer_imag);
        // cudaFree(d_inputDimensions);

        cudaError_t err = cudaFree(d_inputPointer_real);
        // CUSTOM_CUDA_CHECK
        err = cudaFree(d_inputPointer_imag);
        // CUSTOM_CUDA_CHECK
        err = cudaFree(d_inputDimensions);
        // CUSTOM_CUDA_CHECK
    }

    // Methods::constructor
    __host__ CustomGPUObject() {}

    // Methods::constructor::overloaded
    __host__ CustomGPUObject(const mxArray *prhs_in)
    {
        // fetching inputs
        inputPointer_real = (double *)mxGetPr(prhs_in);
        inputPointer_imag = (double *)mxGetPi(prhs_in);
        
        // fetching dimensions
        inputDimensions = (mwSize *)mxGetDimensions(prhs_in);
        numDimensions = (mwSize)mxGetNumberOfDimensions(prhs_in);
        numElements = (mwSize)mxGetNumberOfElements(prhs_in);
        
        // to ensure space allocation takes place
        cuda_space_allocation_flag = 0; 

        // allocating space on CUDA
        if(cuda_space_allocation_flag == 0)
        {
            // mexPrintf("Space allocation begins \n");
            // CUDA_CHECK(cudaMalloc((void**)&d_inputPointer_real, numElements*sizeof(double)));
            // CUDA_CHECK(cudaMalloc((void**)&d_inputPointer_imag, numElements*sizeof(double)));
            // CUDA_CHECK(cudaMalloc((void**)&d_inputDimensions, numDimensions*sizeof(mwSize)));

            cudaError_t err = cudaMalloc((void**)&d_inputPointer_real, numElements*sizeof(double));
            // CUSTOM_CUDA_CHECK
            err = cudaMalloc((void**)&d_inputPointer_imag, numElements*sizeof(double));
            // CUSTOM_CUDA_CHECK
            err = cudaMalloc((void**)&d_inputDimensions, numDimensions*sizeof(mwSize));
            // CUSTOM_CUDA_CHECK

            cuda_space_allocation_flag = 1;
        }
    }

    // Methods::Printing elements
    __host__ void printRealAndImaginary()
    {
        // printing n elements 
        mexPrintf("Elements = [");
        for(int i = 0; i<min(4, (int)numElements); ++i)
        {
            mexPrintf(" %0.2f + %0.2f i, ", inputPointer_real[i], inputPointer_imag[i]);
        }
        mexPrintf("\b\b] \n");
    }
    
    // Methods::Function to populate
    __host__ void PopulateObject(const mxArray *prhs_in)
    {
        // fetching inputs
        inputPointer_real = (double *)mxGetPr(prhs_in);                  // feeding the pointer
        inputPointer_imag = (double *)mxGetPi(prhs_in);
        inputDimensions = (mwSize *)mxGetDimensions(prhs_in);       // fetch dimensions
        numDimensions = (mwSize)mxGetNumberOfDimensions(prhs_in);   // fetch number of dimensions
        numElements = (mwSize)mxGetNumberOfElements(prhs_in);       // fetch number of elements

        // allocating space on CUDA
        if(cuda_space_allocation_flag == 0)
        {
            CUDA_CHECK(cudaMalloc((void**)&d_inputPointer_real, numElements*sizeof(double)));
            CUDA_CHECK(cudaMalloc((void**)&d_inputPointer_imag, numElements*sizeof(double)));
            cuda_space_allocation_flag = 1;
        }
    }

    // Methods::To print dimensions
    __host__ void PrintDimensions()
    {
        mexPrintf("inputDimensions = [");
        for(int i = 0; i<numDimensions; ++i) mexPrintf("%d, ", inputDimensions[i]);
        mexPrintf("\b\b] \n");
    }

    // Methods::function that copies data from host to device
    __host__ void copyFromHostToDevice()
    {
        // copying the real, imaginary and dimensional data from the host to the device
        cudaError_t err = cudaMemcpy(d_inputPointer_real,
                                     inputPointer_real, 
                                     numElements*sizeof(double), cudaMemcpyHostToDevice);
        CUSTOM_CUDA_CHECK

        err = cudaMemcpy(d_inputPointer_imag, \
                         inputPointer_imag, \
                         numElements*sizeof(double), cudaMemcpyHostToDevice);
        // CUSTOM_CUDA_CHECK

        err = cudaMemcpy(d_inputDimensions,
                         inputDimensions, 
                         numDimensions*sizeof(mwSize), 
                         cudaMemcpyHostToDevice);
        // CUSTOM_CUDA_CHECK
    }

    // Methods::function that copies data from device to host
    __host__ void copyFromDeviceToHost()
    {
        // copying the real and imaginary components from the device to the host
        cudaError_t err = cudaMemcpy(inputPointer_real,
                                     d_inputPointer_real, 
                                     numElements*sizeof(double), cudaMemcpyDeviceToHost);
        CUSTOM_CUDA_CHECK

        err = cudaMemcpy(inputPointer_imag,
                         d_inputPointer_imag, 
                         numElements*sizeof(double), cudaMemcpyDeviceToHost);
        // CUSTOM_CUDA_CHECK
    }

};
```