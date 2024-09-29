# Narrowband Beamformer

## Theory
As presented in the previous section, we see that the outputs of each element, in the array, differ in phase. They are made in-phase again by bringing about an artificial delay corresponding to the array position. In this code, this artificial delay is brought in using the following property of the fourier-transform 

$$
x(t - t_0) \leftrightarrow e^{-j \omega t_0} X(\omega)
$$

<div style="margin-top: 6mm;"></div>

So to explain the procedure in a step-manner, it is as follows
<div style="margin-top: -3mm;"></div>

1. The input is fourier-transformed.
2. A weight-vector is defined for a set of equally spaced angles ranging from 0 to 180. 
3. The fourier-transform of the input and the weight-vector is multiplied. 
4. Then the absolute value vs angle is plotted. 

<div style="margin-top: 6mm;"></div>

The result of this particular procedure results in beam-patterns where there is a maxima for the angle that equals the original direction of arrival and for every other angles, an absolute value much smaller than the peak-value is given. 


<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->

## MATLAB Code

<!-- ======================================================================= -->
We first start off by defining the parameters of our uniform-linear-array. 
```Matlab
%% Initializing sensor-array paramters
angle = 60;                 % angle of arrival
signal_frequency = 2000;    % frequency of the arriving signal
Fs = 12800;                 % sampling-frequency of the sensors
Ts = 1/Fs;                  % corresponding time-period
sound_speed = 1500;         % speed of sound in water
num_sensors = 64;           % number of sensors in the sensor-array
```
<!-- ======================================================================= -->
Next, we define some dependent parameters that will be used to simulate the signals that arrive at the sensors in the sensor-array.
```Matlab
SNR = 1;                                % SNR of the received signal
N = 256;                                % number of samples we'll be simulating
t = (0:N-1)*Ts;                         % time-periods
lambda = sound_speed/signal_frequency;  % wavelength of arriving signal
x = lambda/2;                           % sensor-spacing
d = x*cosd(angle)/sound_speed;          % time-delay between sensors
signalMatrix = zeros(N,num_sensors);    % declaring the signal-matrix
```
<!-- ======================================================================= -->
Next, we simulate the signals that would arrive at the signal array for the current parameters and conditions. Since we have privileged information about the nature of the signal, we directly simulate it using the calculated delays. And this simulation is done element-by-element. 
```Matlab
%% Bringing the natural delay
y = sin(2*pi*signal_frequency*t);
tend = (N-1)*Ts;
xaxis = linspace(0, tend, N);

%% Building Signal
for i = 1:num_sensors
    signalMatrix(:,i) = sin(2*pi*signal_frequency*(t+(i-1)*d));
end
```
<!-- ======================================================================= -->
Next, we define the set of angles for which we'll be beamforming. We choose to beamform for 512 equally spaced angles in the range of $[0,180]$. Note that beamforming for one angle is independent of calculations performed for others. Thus, beamforming is highly parallelizable. This means that the more number of angles we want to beamform for, the better GPU shines at the task. So feel free to change the number of angles for which we're beamforming below and see the beampattern get more finer. After defining the set of angles for which we're beamforming, we calculate the delays associated with those angles using the expression obtained before. 
```Matlab
%% Beamforming parameters
anglesToBeamform = linspace(0, 180, 512);
delaysToBeamform = x*cosd(anglesToBeamform)/sound_speed;
```
<!-- ======================================================================= -->
Next, we compile the CUDA code and call the function with the arguments we setup. Note that this compilation doesn't need to be done each time and just needs to be once after any changes has been made to the CUDA code. I've thrown this in here to make things easier. And it is generally a good idea to throw in the compilation step in here (provided it doesn't take too much time to compile) because any kind of errors will be immediately brought to our attention. Back to the task, after the function has been called, the returned signal will be a complex signal. 
```Matlab
%% compiling the mex-code
mexcuda fdb_v2_structure.cu

%% calling the function
signalMatrix_fft = fft(signalMatrix);
beamformedMatrix = fdb_v2(signalMatrix_fft, ...
                            delaysToBeamform, ...
                            Fs);
```
<!-- ======================================================================= -->
Now that we've obtained the beamformed complex signals for each angle, we check the amplitude of the fourier-coefficient corresponding to the frequency of the signal.The angle which has the highest coefficient corresponds to the direction of arrival. The beam-pattern is a plot with angles along the x-axis and the $\log$ of the coefficient along the y-axis. We create and plot the beam-pattern for this particular algorithm and parameters. 


```Matlab
%% Plotting beampattern
index_of_interest = 1 + ...
                    floor(size(signalMatrix,1)*signal_frequency/Fs);
rowOfInterest = abs(beamformedMatrix(index_of_interest,:));
figure(1);
subplot(2,1,1); plot(anglesToBeamform, rowOfInterest);
subplot(2,1,2); plot(anglesToBeamform, log(rowOfInterest));

```

### Final Matlab Code
<!-- ======================================================================= -->
Putting it all together we get the following Matlab code

```matlab
%{
   Aim: Implementing directional frequency-domain beamforming
%}

%% basic setup
clc; clear;

%% Initializing sensor-array paramters
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

%% compiling the mex-code
mexcuda fdb_v2_structure.cu

%% calling the function
signalMatrix_fft = fft(signalMatrix);
beamformedMatrix = fdb_v2(signalMatrix_fft, ...
                            delaysToBeamform, ...
                            Fs);

%% Plotting beampattern
index_of_interest = 1 + floor(size(signalMatrix,1)*signal_frequency/Fs);
rowOfInterest = abs(beamformedMatrix(index_of_interest,:));
figure(1);
subplot(2,1,1); plot(anglesToBeamform, rowOfInterest);
subplot(2,1,2); plot(anglesToBeamform, log(rowOfInterest));
```










<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
## CUDA Code

In the CUDA code, we need to define two functions: gateway function and kernel-definition. 

### Kernel Definition
The kernel takes in the following set of parameters. 
1. Pointer to the array containing the dimensions of the signal-matrix, in the GPU global-memory. 
2. Number of angles to beamform
3. Pointer to the real-part of signal-matrix, in the GPU global-memory
4. Pointer to the imaginary-part of the signal-matrix, in the GPU global-memory. 
5. Pointer to the real-part of the beamformed-matrix, in the GPU global-memory
6. Pointer to the imaginary-part of the beamformed-matrix, in the GPU global-memory. 
7. Pointer to the array containing the delays, in the GPU global-memory. 
8. The sampling-frequency

```C
// kernels
__global__ void beamformKernel(mwSize *signalMatrix_d_inputDimensions,
                                int num_angles_to_beamform,
                                double *signalMatrix_d_inputPointer_real,
                                double *signalMatrix_d_inputPointer_imag,
                                double *beamformedMatrix_d_inputPointer_real,
                                double *beamformedMatrix_d_inputPointer_imag,
                                double *delays_d_inputPointer,
                                double sampling_frequency)
{
    ...
}
```

<!-- ======================================================================= -->
In this particular operation, we use the shared-memory. And since we're using dynamic shared-memory for this task, we declare the pointer to the shared-memory in the following manner. 
```C
    // declaring the shared memory
    extern __shared__ double sharedMem[];
```

<!-- ======================================================================= -->
In our kernel design, each block is assigned the responsibility of delaying the signal from a particular sensor by the appropriate delay corresponding to the angle for which we're beamforming and the sensor position. And each thread is assigned the responsibility of carrying out sample-level complex multiplications. And the grid-configuration is in such a way that all the threads-blocks in a particular depth (third-dimension) is responsible for carrying out the beamforming for a particular angle. So the threads are mapped to the the sample it is responsible for and the blocks are assigned to each sensor. 
Next, we calculate the input the particular thread should work with. Since the data-structure containing the input-signal is a matrix and since matrices are stored linearly in the GPU global-memory, we also produce the linear mappings into the matrix.

```C
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
```

<!-- ======================================================================= -->
Next, we initialize some variables for some better readability. We infer the number of sensors from the input-matrix since the dimensionalit of the matrix along axis = 1 gives us the number of sensors. As mentioned before, since the blocks at each "level" are assigned the responsibility of carrying out calculations for an angle, we obtain the angle this particular thread is responsible for, in the following manner. 
```C
    // miscellaneous
    const int num_sensors = (int)signalMatrix_d_inputDimensions[1];
    const int current_angle = blockIdx.z;
```

<!-- ======================================================================= -->
Next, we produce the element-wise complex multiplication that comes with delaying signals. It contains the following set of steps
1. The first step is to check the validity of the threads. Note that this is an important step because it is often the case that the number of threads launched is more than the number of threads required for the task. Thus validity checks is always highly recommended. The validity checks validity of current-sensor, validity of current-sample and the validity of the current-angle. 
2. After validity checks, we obtain the number of samples in the signal, calculated from the dimensions of the input-matrix. 
3. Next, we calculate the delaying value
4. We then produce the real-part of the complex multiplication and imaginary-part of the complex multiplication. 
5. The results is then stored to the shared-memory. 
6. This is followed by thread syncing so that the accumulation procedure that follows won't have to work with incomplete values. 
```C
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
        const double current_real = signalMatrix_d_inputPointer_real[linearIndex2D_input];
        const double current_imag = signalMatrix_d_inputPointer_imag[linearIndex2D_input];

        // obtaining resulting real and imaginary parts
        const double result_real_part = current_real*cos(wt) + current_imag*sin(wt);
        const double result_imag_part = current_imag*cos(wt) - current_real*sin(wt);

        // copying to the shared memory
        sharedMem[2*linearIndex2D_dim1] = result_real_part;
        sharedMem[1 + 2*linearIndex2D_dim1] = result_imag_part;
    }

    // syncing threads
    __syncthreads();

```

<!-- ======================================================================= -->
Now that the element-wise multiplications for the delaying has been completed, we assign the first thread to accumulate the values. We ensure that just one thread does it by first creating a gate that only the first thread from the block can calculate the value. The accumulation is done for both the real-part and imaginary part independently and finally stored into the space allocated for the beamformed matrix. 
```C
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
```


<!-- ====================================================================== -->
Putting it together, we get the following kernel definition s
```C

// kernels
__global__ void beamformKernel(mwSize *signalMatrix_d_inputDimensions,
                                int num_angles_to_beamform,
                                double *signalMatrix_d_inputPointer_real,
                                double *signalMatrix_d_inputPointer_imag,
                                double *beamformedMatrix_d_inputPointer_real,
                                double *beamformedMatrix_d_inputPointer_imag,
                                double *delays_d_inputPointer,
                                double sampling_frequency)
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
        const double current_real = signalMatrix_d_inputPointer_real[linearIndex2D_input];
        const double current_imag = signalMatrix_d_inputPointer_imag[linearIndex2D_input];

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
```



### Gateway function
In the gateway function, the first thing we check is the validity of the inputs that were passed by the matlab script calling our function. This checks usually involve the number of inputs, number of expected outputs, the data-type of the inputs and the dimensionality of the inputs. Note that this is super important to do but we're skipping it here since, from the previous examples, you must be super familiar with how to write these. So we jump straight into fetching the inputs. 

The received inputs are encapsulated using objects of the class, *CustomGPUObject*. The inputs are then made available in the GPU global-memory using the class method, *copyFromHostToDevice*. 
```C

// main function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Fetching inputs
    CustomGPUObject signalMatrix(prhs[0]);
    CustomGPUObject delayMatrix(prhs[1]);
    double sampling_frequency = (double)mxGetScalar(prhs[2]);

    // Sending the data and input-dimensions to the GPU
    signalMatrix.copyFromHostToDevice();
    delayMatrix.copyFromHostToDevice();

    ...
}
```

<!-- ======================================================================= -->
Next, we allocate space for the output. To do this, first we calculate the output-dimensions based on the parameters we received from the inputs. Once the output-dimensions have been obtained, we create a Matlab matrix using the function, *mxCreateNumericArray()*. The matrix created using this method is encapulsated using an object of the class, *CustomGPUObject*. We then allocate space for the output in the GPU global-memory using the class-method, *copyFromHostToDevice()*. 
```C
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
```

<!-- ======================================================================= -->
This is followed by setting up the launch-configuration parameters: block-configuration, grid-configuration and shared-memory size. In our kernel definition, since each thread is assigned the responsibility of carrying out element-wise multiplication for a signal, the number of threads in a block is equal to the number of samples recorded by a sensor. Since each block is assigned the responsibility for dealing with one sensor, the number of blocks are chosen in such a way that, for one angle, the number of blocks assigned to it is greater than or equal to the number of sensors. The number of blocks along the third dimension is equal to the number of angles for which, we must beamform. The shared-memory size is calculated to be twice the number of elements required to store a signal obtained by a sensor. This is because we're calculating and storing for both real and imaginary numbers. AFter this, the kernel is launched into the default stream using the launch-configuration parameters and the function arguments. Since launches to the default stream are block, we copy the results back from the device memory space to the host-memory space right after the function definition. 
```C
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
                             sampling_frequency);


    // copying the data from the device to the host
    beamformedMatrix.copyFromDeviceToHost();
```


<!-- ======================================================================= -->
### Final CUDA Code
Putting together the kernel-definition and gateway function, we get the following. 


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
                                double sampling_frequency)
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
        const double current_real = signalMatrix_d_inputPointer_real[linearIndex2D_input];
        const double current_imag = signalMatrix_d_inputPointer_imag[linearIndex2D_input];

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

    // Sending the data and input-dimensions to the GPU
    signalMatrix.copyFromHostToDevice();
    delayMatrix.copyFromHostToDevice();

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
                        sampling_frequency);


    // copying the data from the device to the host
    beamformedMatrix.copyFromDeviceToHost();

    // shutting down
    cudaDeviceSynchronize();
    cudaDeviceReset();
}     
```

<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->
<!-- B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B=B -->

## References
- "Discrete-time Beamforming: Frequency-domain Beamforming." Wikipedia, Wikimedia Foundation, [Link](https://en.wikipedia.org/wiki/Discrete-time_beamforming#Frequency-domain_beamforming). Accessed 29 Sept. 2024.
- MathWorks. “mxCreateNumericArray.” MATLAB API Reference, MathWorks, [Link](www.mathworks.com/help/matlab/apiref/mxcreatenumericarray.html). Accessed 26 Sept. 2024.