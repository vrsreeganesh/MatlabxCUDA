# Narrowband Beamformer with Sensor-Weighing

## Theory 
Traditional beamforming has some disadvantages. The first, second and third order sidelobes are 13.5, 18 and 21 dB below the peak-level of the main-lobe. This means that strong signals will be detected throguh the sidelobes of adjacent beams, in addition to the detection in the main-lobe. This is undesirable since it results in ambiguity, and additional false-signals can complicate downstream processing [1]. 

The aim, therefore, is to produce the narrowest possible main-lobe consistent with some reasonable level of sidelobes. One method to achieve this is by using [chebyshev window](https://en.wikipedia.org/wiki/Window_function). This means that the output of each element is scaled by the window-coefficients. The window, essentially, minimizes the chebyshev norm of the side-lobes for a given main-lobe width. 


## Matlab Code

<!-- =================================================== -->
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
<!-- =================================================== -->
Next, we define some dependent parameters that will be used to simulate the signals that arrive at the sensors in the sensor-array
```Matlab
SNR = 1;
N = 256;
t = (0:N-1)*Ts;
lambda = sound_speed/signal_frequency;
x = lambda/2;
d = x*cosd(angle)/sound_speed;
signalMatrix = zeros(N,num_sensors);
```
<!-- =================================================== -->
Next, we simulate the signals received by the elements for this particular configuration/setup. The signal is a simple sine-wave and the sensor outputs are created under the assumption that the signals are coming from infinity, which basically means that the relative delays are purely dependent on the angles. 

```matlab
%% Bringing the natural delay
y = sin(2*pi*signal_frequency*t);
tend = (N-1)*Ts;
xaxis = linspace(0, tend, N);

%% Building Signal
for i = 1:num_sensors
    signalMatrix(:,i) = sin(2*pi*signal_frequency*(t+(i-1)*d));
end
```
<!-- =================================================== -->
Next, we define the set of angles for which we’ll be beamforming. We choose to beamform for 512 equally spaced angles in the range of 
[0, 180]. Note that beamforming for one angle is independent of calculations performed for others. Thus, beamforming is highly parallelizable. This means that the more number of angles we want to beamform for, the better GPU shines at the task. So feel free to change the number of angles for which we’re beamforming below and see the beampattern get more finer. After defining the set of angles for which we’re beamforming, we calculate the delays associated with those angles using the expression obtained before.
```matlab
%% Beamforming parameters
anglesToBeamform = linspace(0, 180, 512);
delaysToBeamform = x*cosd(anglesToBeamform)/sound_speed;
```
<!-- =================================================== -->
Next, we setup the window in the following manner 
```matlab
%% Building Sensor Weights
cheb_window = chebwin(num_sensors, 30);
```
<!-- =================================================== -->
Next, we compile the CUDA code we've written for this particular operation and call the associated mex-function with the necessary parameters in the following manner. 
```matlab
%% compiling the mex-code
mexcuda fdb_SensorWeights.cu
%% calling the function
signalMatrix_fft = fft(signalMatrix);
beamformedMatrix = fdb_SensorWeights(signalMatrix_fft, ...
                                     delaysToBeamform, ...
                                     Fs, ...
                                     cheb_window);
```

<!-- =================================================== -->
Once the results are obtained, we plot the beampattern with the windowing and without the windowing in the following manner 
```matlab
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

### Final Matlab Code
Putting it all-together, we get the following 
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




<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
<!-- A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A=A= -->
## CUDA Code
In the CUDA code, we need to define two functions: gateway function and kernel-definition.

### Kernel Definition 
The kernel takes in the following argumentss 

1. Pointer to array containing dimensions of signal matrix, in the GPU-memory.
2. Number of angles to beamform
3. Pointer to the real-part of signal-matrix, in the GPU global-memory
4. Pointer to the imaginary-part of signal-matrix, in the GPU global-memory
5. Pointer to the real-part of the beamformed matrix, in the GPU global-memory
6. Pointer to the imaginary-part of the beamformed matrix, in the GPU global-memory. 
7. Pointer to array containing delays, in the GPU global-memory
8. Sampmling frequency of the sensor arrays
9. Pointer to the array containing window coefficients, in the GPU global-memory. 

```C
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
    ...
}
```

<!-- =================================================== -->
In this operation, we'll be using the shared-memory. And since we're using dynamic shared-memory for the task, we declare pointer to the shared-memory in the following manner 
```C
    // declaring the shared memory
    extern __shared__ double sharedMem[];
```
<!-- =================================================== -->
In our kernel design, each block is assigned the responsibility of delaying the signal from a particular sensor by the appropriate delay corresponding to the angle for which we’re beamforming and the sensor position. And each thread is assigned the responsibility of carrying out sample-level complex multiplications. And the grid-configuration is in such a way that all the threads-blocks in a particular depth (third-dimension) is responsible for carrying out the beamforming for a particular angle. So the threads are mapped to the the sample it is responsible for and the blocks are assigned to each sensor. Next, we calculate the input the particular thread should work with. Since the data-structure containing the input-signal is a matrix and since matrices are stored linearly in the GPU global-memory, we also produce the linear mappings into the matrix.
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

<!-- =================================================== -->
Next, we initialize some variables for some better readability. We infer the number of sensors from the input-matrix since the dimensionalit of the matrix along axis = 1 gives us the number of sensors. As mentioned before, since the blocks at each “level” are assigned the responsibility of carrying out calculations for an angle, we obtain the angle this particular thread is responsible for, in the following manner.
```C
    // miscellaneous
    const int num_sensors = (int)signalMatrix_d_inputDimensions[1];
    const int current_angle = blockIdx.z;
```

<!-- =================================================== -->
Next, we produce the element-wise complex multiplication that comes with delaying signals. The steps for this are as follows
1. Check the validity of the threads. Note that this is important because, it is often the case that, the number of threads launched is greater than the number of threads required for the job. Thus, lack of validity checks can result in unexpected behaviour. Thus, validity checks are highly recommended. In our context, we check the validity of the linear-indices we're fetching and if the current-angle we're dealing with is in the angles we're beamforming for. 
2. Infer the number of samples in the signal
3. Calculate the delaying value
4. Produce the real and imaginary part of the complex-multiplication. Note that this is where our code conceptually deviates from that of rectangular windowing. We see that the real and imaginary components of the signal are scaled by the window-coefficient corresponding to the sensor, which the signal belongs to, before the complex multiplication. 
5. Store the results
6. This is followed by thread-syncing to ensure that the rest of the code is only implemented after all the threads reach this checkpoint. 

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

    __syncthreads();
```
<!-- =================================================== -->
Now that the element-wise multiplications, for the delaying, has been completed, we assign the task of accumulating to the first thread. We ensure that just one thread does it by first creating a gate that only the first thread in a block can pass through. The accumulation for the real and imaginary parts are carried out independently and then stored into the memory allocated for the beamformed matrix. 

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

    // syncing threads
    __syncthreads();
```

<!-- =================================================== -->
Putting together the kernel definition, we get the following 
```C
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
```


## Gateway Function
In the gateway function, the first step is ensure validity of inputs, and **expected** outputs, passed from Matlab. The checks usually involve checking the number of inputs, dimensionality of the input, data-type of the input and the expected number of outputs. It is imperative that this section be present in all gateway functions. However, we're skipping it here since input-verification has been presented in previous chapters and I'd like to keep the code, light, here. 

So the inputs received from Matlab are encapsulated using objects of the class, *CustomGPUObject*. These inputs are then made available in the GPU global-memory using the class method, *copyFromHostToDevice*. Our task requires the sensor-array signals, the matrix containing the delays to be given for each angle, sampling-frequency and the array containing the window-coefficients. We check to ensure that the number of sensor-coefficients are the same as the number of sensor-outputs that is available to us. Also, note that we're not encapsulating the sampling-frequency in our object because scalars can be made easily available in the GPU global-memory by just passing them by value. 
```C
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

    ...
}
```
<!-- =================================================== -->
Next, we allocate space for the output of our procedure. The first step for this is to calculate the output-dimensions based on the parameters we've received and inferred. We create the Matlab-matrix using the function, *mxCreateNumericArray()*. This matrix is then encapsulated using an object of class, *CustomGPUObject*. We then make space for the output in the GPU-global memory using the class method, *copyFromHostToDevice()*. The reader is directed to the class-definition to  better understand what the method does. 

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
    beamformedMatrix.copyFromHostToDevice();
```

<!-- ======================================================================= -->
This step is followed by setup for the launch. We initialize the launch-configuration parameters: block-configuration, grid-configuration and shared-memory size. 
- In our kernel definition, since each thread is assigned the responsibility of carrying out element-wise multiplication for a sensor-output, the number of threads in a block is equal to the number of samples recorded by a sensor. 
- Since each block is assigned the responsibility for dealing with one sensor, the number of blocks are assigned such that, for one angle, the number of blocks are greater than or equal to the number of sensors. The number of blocks along the third-dimension is equal to the number of angles for which we're beamforming.
- The shared-memory size is assigned as twice the number of elements required to store a sensor-signal. This is because we're calculating and storing for both real and imaginary numbers.

After initialization of launch-configuration parameters, the kernel is launched into the default stream using the launch-configuration parameters and the function arguments. Since launches to default streams are blocking, we write the line for copying the results back, right after this section. 

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
                        sampling_frequency,
                        sensorWindow.d_inputPointer_real);

    // copying the data from the device to the host
    beamformedMatrix.copyFromDeviceToHost();
```
<!-- ======================================================================= -->
Putting it together, the gateway function is as follows 
```C
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

<!-- ======================================================================= -->

### Final CUDA Code
Putting together the kernel-definition, gateway-function and some miscellaneous code sections, we finally get the following 

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