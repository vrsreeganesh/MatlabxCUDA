# Narrowband Beamformer with Sensor-Weighing



Traditional beamforming (ie, rectangular windowing) brings along with it significant disadvantages. The first-order, second-order and third-order sidelobes are respectively only 13.5, 18 and 21 dB below the peak level of the main lobe. Strong signals will be detected through the sidelobes of adjacent beams as well as, correctly, in the main lobe of the beam at the bearing of the signal. The resultant bearing ambiguity and additional, false, signals complicate all further
processes [1].


The aim, therefore, should be to produce the narrowest possible main lobe consistent with some reasonable level of sidelobes. One such method is [chebyshev window](https://en.wikipedia.org/wiki/Window_function). Here, we multiply the amplitude of each element’s output with shading coefficients obtained from the chebwin function in matlab.


## Matlab Code

We start off by defining the parameters of our uniform linear-array

```Matlab
% sensor array parameters
angle = 60; 
signal_frequency = 2000;
Fs = 12800; 
Ts = 1/Fs;
sound_speed = 1500;
num_sensors = 64;
```


```matlab
%{
=================================================================================
Aim: Implementing directional frequency-domain beamforming
=================================================================================
%}

%% basic setup
clc; clear;

%% Initializing variables
angle = 60; 
signal_frequency = 2000;
Fs = 12800; 
Ts = 1/Fs;
sound_speed = 1500;
num_sensors = 64;

SNR = 1;
N = 256;
t = (0:N-1)*Ts;
lambda = sound_speed/signal_frequency;
x = lambda/2;
d = x*cosd(angle)/sound_speed;
signalMatrix = zeros(N,num_sensors);

%% Bringing the natural delay
y = sin(2*pi*signal_frequency*t);
tend = (N-1)*Ts;
xaxis = linspace(0, tend, N);

%% Building Signal
for i = 1:num_sensors
    signalMatrix(:,i) = sin(2*pi*signal_frequency*(t+(i-1)*d));
end

%% Beamforming parameters
anglesToBeamform = linspace(0, 180, 512);
delaysToBeamform = x*cosd(anglesToBeamform)/sound_speed;

%% Building Sensor Weights
cheb_window = chebwin(num_sensors, 30);

%% compiling the mex-code
mexcuda fdb_SensorWeights.cu

%% calling the function
signalMatrix_fft = fft(signalMatrix);
beamformedMatrix = fdb_SensorWeights(signalMatrix_fft, ...
                                        delaysToBeamform, ...
                                        Fs, ...
                                        cheb_window);

%% Plotting beampattern with windowing
index_of_interest = 1 + floor(size(signalMatrix,1)*signal_frequency/Fs);
rowOfInterest = abs(beamformedMatrix(index_of_interest,:));
figure(1);
subplot(2,2,1); plot(anglesToBeamform, rowOfInterest); title("linear-Beampattern, windowing")
subplot(2,2,3); plot(anglesToBeamform, log(rowOfInterest)); title("log-beampattern, windowing")

%% Plotting without windowing
beamformedMatrix = fdb_SensorWeights(signalMatrix_fft, ...
                                        delaysToBeamform, ...
                                        Fs, ...
                                        ones(size(cheb_window)));

%% Plotting beampattern
index_of_interest = 1 + floor(size(signalMatrix,1)*signal_frequency/Fs);
rowOfInterest = abs(beamformedMatrix(index_of_interest,:));
subplot(2,2,2); plot(anglesToBeamform, rowOfInterest); title("linear-beampattern, non-windowing");
subplot(2,2,4); plot(anglesToBeamform, log(rowOfInterest)); title("linear-beampattern, non-windowing");
```



## CUDA Code

```C
/*
=================================================================================
Aim: Directional frequency-domain beamforming using CUDA
=================================================================================
*/

// header-files
#include "mex.h"
#include "booktools.h"

#define PI 3.14159265

// kernels
__global__ void beamformKernel(mwSize *signalMatrix_d_inputDimensions,
                                int num_angles_to_beamform,
                                double *signalMatrix_d_inputPointer_real,
                                double *signalMatrix_d_inputPointer_imag,
                                double *beamformedMatrix_d_inputPointer_real,
                                double *beamformedMatrix_d_inputPointer_imag,
                                double *delays_d_inputPointer,
                                double sampling_frequency,
                                double *sensorWindow_d_inputPointer_real)
{
    // declaring the shared memory
    extern __shared__ double sharedMem[];

    // address along dimension 1
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int linearIndex2D_dim1 = tidx + tidy*blockDim.x;

    // address along dimension 0
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int linearIndex2D_dim0 = bidx + bidy*gridDim.x;

    // linear addressing into the inputs
    const int linearIndex2D_input = \
        linearIndex2D_dim0 + \
        linearIndex2D_dim1*signalMatrix_d_inputDimensions[0];

    // miscellaneous
    const int num_sensors = (int)signalMatrix_d_inputDimensions[1];
    const int current_angle = blockIdx.z;

    // weighing and storing to the shared matrix
    if(linearIndex2D_dim0 < signalMatrix_d_inputDimensions[0] && \
        linearIndex2D_dim1 < signalMatrix_d_inputDimensions[1] && \
        current_angle < num_angles_to_beamform)
    {
        // weights
        const int N = (int)signalMatrix_d_inputDimensions[0];
        double wt = 2*PI*(double)linearIndex2D_dim0*sampling_frequency/N*\
                            (double)linearIndex2D_dim1*delays_d_inputPointer[blockIdx.z];

        // obtaining current real and imaginary part
        const double current_real = \
            signalMatrix_d_inputPointer_real[linearIndex2D_input] * \
            sensorWindow_d_inputPointer_real[linearIndex2D_dim1];
        const double current_imag = \
            signalMatrix_d_inputPointer_imag[linearIndex2D_input] * \
            sensorWindow_d_inputPointer_real[linearIndex2D_dim1];

        // obtaining resulting real and imaginary parts
        const double result_real_part = current_real*cos(wt) + current_imag*sin(wt);
        const double result_imag_part = current_imag*cos(wt) - current_real*sin(wt);

        // copying to the shared memory
        sharedMem[2*linearIndex2D_dim1] = result_real_part;
        sharedMem[1 + 2*linearIndex2D_dim1] = result_imag_part;
    }

    // syncing threads
    __syncthreads();

    // Getting the first element to adding them up
    if(linearIndex2D_dim0 < signalMatrix_d_inputDimensions[0] && \
        linearIndex2D_dim1==0 && \
        current_angle < num_angles_to_beamform)
    {

        // getting the first element of each block to add things up
        double accuSum_real = 0;
        double accuSum_imag = 0;
        for(int i = 0; i<num_sensors; ++i)
        {
            accuSum_real = accuSum_real + sharedMem[2*i];
            accuSum_imag = accuSum_imag + sharedMem[2*i + 1];
        }

        // linear indexing to output tensor
        const int linearIndex2D_output = linearIndex2D_dim0 + blockIdx.z*signalMatrix_d_inputDimensions[0];

        // copying:: verified
        beamformedMatrix_d_inputPointer_real[linearIndex2D_output] = accuSum_real;
        beamformedMatrix_d_inputPointer_imag[linearIndex2D_output] = accuSum_imag;
    }

    // syncing threads
    __syncthreads();

}


// main function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Fetching inputs
    CustomGPUObject signalMatrix(prhs[0]);
    CustomGPUObject delayMatrix(prhs[1]);
    double sampling_frequency = (double)mxGetScalar(prhs[2]);
    CustomGPUObject sensorWindow(prhs[3]);

    // A minor check
    if(sensorWindow.numElements != signalMatrix.inputDimensions[1])
        mexErrMsgTxt("Length of window is expected to be number of sensors \n");

    // Sending the data and input-dimensions to the GPU
    signalMatrix.copyFromHostToDevice();
    delayMatrix.copyFromHostToDevice();
    sensorWindow.copyFromHostToDevice();

    // preparing to create output
    mwSize *beamformedMatrix_outputDimensions = (mwSize *)malloc(2*sizeof(mwSize));
    beamformedMatrix_outputDimensions[0] = (mwSize)signalMatrix.inputDimensions[0];
    beamformedMatrix_outputDimensions[1] = (mwSize)delayMatrix.numElements;

    // creating an output
    plhs[0] = mxCreateNumericArray(((mwSize)2),
                                    beamformedMatrix_outputDimensions,
                                    mxDOUBLE_CLASS,
                                    mxCOMPLEX);

    CustomGPUObject beamformedMatrix(plhs[0]);

    // calling the function
    const int num_threads = 1 + (int)sqrt(signalMatrix.inputDimensions[1]);
    const int num_blocks = 1 + (int)sqrt(signalMatrix.inputDimensions[0]);
    dim3 blockConfiguration(num_threads, num_threads);
    dim3 gridConfiguration(num_blocks, num_blocks, (int)delayMatrix.numElements);

    beamformKernel<<<gridConfiguration,
                        blockConfiguration,
                        2*signalMatrix.inputDimensions[1]*sizeof(double)>>>
                        (signalMatrix.d_inputDimensions,
                        delayMatrix.numElements,
                        signalMatrix.d_inputPointer_real,
                        signalMatrix.d_inputPointer_imag,
                        beamformedMatrix.d_inputPointer_real,
                        beamformedMatrix.d_inputPointer_imag,
                        delayMatrix.d_inputPointer_real,
                        sampling_frequency,
                        sensorWindow.d_inputPointer_real);

    // copying the data from the device to the host
    beamformedMatrix.copyFromDeviceToHost();

    // shutting down
    cudaDeviceSynchronize();
    cudaDeviceReset();
}
```

## References
- “Window Function.” Wikipedia: The Free Encyclopedia, Wikimedia Foundation, 26 Sept. 2024, [Link](en.wikipedia.org/wiki/Window_function). Accessed 26 Sept. 2024.
- Smith, Julius O. “Dolph-Chebyshev Window.” Spectral Audio Signal Processing, DSPRelated.com, [Link](www.dsprelated.com/freebooks/sasp/Dolph_Chebyshev_Window.html). Accessed 26 Sept. 2024.