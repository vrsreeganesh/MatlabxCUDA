# Signal Simulation from Image



## Matlab

```Matlab
%{
=================================================================================
Aim: Simulate Signals From an Image
=================================================================================
%}

%% Basic setup
clc; clear;

%% Compiling CUDA code 
mexcuda signal_simulation_imaging_structure.cu

%% Sensor Array Parameters
sensorarray_origin = [0;0];            % The location of the first sensor in the sensor-array
sensor_spacing = 1e-1;                 % the distance between sensors in the ULA
sensorarray_direction = [1;0];         % The direction from first sensor to rest of the sensors
sensorarray_direction = sensorarray_direction/(vecnorm(sensorarray_direction));
sensorarray_direction = sqrt(sensor_spacing)*sensorarray_direction;
num_sensors = 256;                     % number of sensors in the ULA
emitter_location = sensorarray_origin; % location of ping emitter

%% Signal Parameters
c = 1500;                                                   % Speed of Sound in Water
signal_frequency = 100;                                     % Frequency of Transmitted Sine-wave
recording_time = 2e-1;                                      % Duration of Recording
sampling_frequency = 16e3;                                  % Sampling Frequency of Hydrophones
transmissionDuration = 1e-2;                                % Duration of transmission
timeArray = 0:(1/sampling_frequency):transmissionDuration;  % sampling intervals
transmissionSignal = chirp(timeArray, ...
                            100, ...
                            timeArray(end), ...
                            sampling_frequency/2);
max_range = transmissionDuration*c/2;


%% Buidling Signal Target
inputImage = phantom(1024);

inputImage = zeros(size(inputImage));
% inputImage(end-100:end, end/2:end/2+20) = 1;
% inputImage(end,1) = 1;

% inputImage(1, 1) = 1;
% inputImage(end, 1) = 1;
% inputImage(1, end) = 1;
% inputImage(end, end) = 1;

inputImage(end/2, end/2) = 1;

% top_left_coordinate = [-max_range*0.8;1];
% bottom_right_coordinate = [max_range*0.8; max_range*1];

top_left_coordinate = [0;-1*max_range];
bottom_right_coordinate = [max_range; 1*max_range];

%% Calling Signal Simulation Script
tic
[distanceMatrix, ...
    amplitudeMatrix, ...
    signalMatrix] = ...
    signal_simulation_imaging_structure(sensorarray_origin, ...
                                        sensorarray_direction, ...
                                        sensor_spacing, ...
                                        num_sensors, ...
                                        emitter_location, ...
                                        inputImage, ...
                                        top_left_coordinate, ...
                                        bottom_right_coordinate, ...
                                        c, ...
                                        signal_frequency, ...
                                        recording_time, ...
                                        sampling_frequency, ...
                                        transmissionSignal);
toc


%% Plotting sensor-wise distances
figure(1)
dynamicRangeAdjustedAmplitudeMatrix = log(1+amplitudeMatrix);
imagesc(dynamicRangeAdjustedAmplitudeMatrix); title("Pre-Convolution Output");

figure(2);
subplot(3,1,1); plot(signalMatrix(:, 1)); title("Signal Received By First Sensor");
subplot(3,1,2); plot(signalMatrix(:, 2)); title("Signal Received By Middle Sensor");
subplot(3,1,3); plot(signalMatrix(:, 3)); title("Signal Received By End Sensor");

```

## CUDA


```C
/*
================================================================================
Aim: Demonstrate Signal Simulation from a High-frequency Image
================================================================================
*/ 

// header-file
#include "mex.h"
#include "booktools.h"

// kernels
__global__ void PerPixel(mwSize *d_inputDimensions_Image,
                            double *d_bottom_left_corner,
                            double *d_top_right_corner,
                            double *d_sensor_origin,
                            double *d_sensor_direction,
                            double *d_outputPointer,
                            double *d_inputPointer_NumSensors,
                            double *d_inputPointer_SoundSpeed,
                            double *d_inputPointer_EmitterLocation,
                            double *d_inputPointer_SamplingFrequency,
                            double *d_inputPointer_InputImage,
                            double *d_inputPointer_RecordingTime,
                            double *d_outputPointer_AmplitudeMatrix)
{
    // getting address
    const int tidx_pixel = blockIdx.x;
    const int tidy_pixel = blockIdx.y;
    const int linearIndex2D_pixel = tidx_pixel + tidy_pixel*d_inputDimensions_Image[0];

    // Calculating inter-pixel spacing
    const double pixel_x_spacing = \
        (double)((d_top_right_corner[0] - d_bottom_left_corner[0])/((double)d_inputDimensions_Image[0]-1));
    const double pixel_y_spacing = \
        (double)((d_top_right_corner[1] - d_bottom_left_corner[1])/((double)d_inputDimensions_Image[1]-1));

    // Getting Pixel Characteristics
    const double pixel_x_coordinate = d_bottom_left_corner[0] + \
                                        (tidx_pixel-1)*pixel_x_spacing;
    const double pixel_y_coordinate = d_bottom_left_corner[1] + \
                                        (tidy_pixel-1)*pixel_y_spacing;

    // getting address
    const int tidx_sensor = threadIdx.x;
    const int tidy_sensor = threadIdx.y;
    const int tidz_sensor = threadIdx.z;
    const int linearIndex2D_sensor = tidx_sensor + \
                                        tidy_sensor*blockDim.x + \
                                        tidz_sensor*blockDim.x*blockDim.y;

    const int sensor_number = linearIndex2D_sensor;

    // Getting Sensor Characteristics
    const double sensor_x_coordinate = d_sensor_origin[0] + \
                                        (sensor_number-1)*d_sensor_direction[0];
    const double sensor_y_coordinate = d_sensor_origin[1] + \
                                        (sensor_number-1)*d_sensor_direction[1];

    // Calculating Distance Between Sensor and Pixel
    const double distance_between_pixel_and_sensor = \
        (double)sqrt(pow((sensor_x_coordinate-pixel_x_coordinate),2) + \
                        pow((sensor_y_coordinate-pixel_y_coordinate),2));

    // Calculating Distance Between Transmitter and Pixel
    const double distance_between_pixel_and_transmitter = \
        (double)sqrt(pow((sensor_x_coordinate - d_inputPointer_EmitterLocation[0]),2) +
                        pow((sensor_y_coordinate - d_inputPointer_EmitterLocation[1]),2));

    // Writing the Value
    if(tidx_pixel<d_inputDimensions_Image[0] && \
        tidy_pixel<d_inputDimensions_Image[1] && \
        sensor_number < (int)d_inputPointer_NumSensors[0])
    {
        // finding index to write to
        const int linearIndex2D_Output = \
            linearIndex2D_pixel + \
            sensor_number*d_inputDimensions_Image[0]*d_inputDimensions_Image[1];

        // finding time
        const double distance_travelled = distance_between_pixel_and_transmitter + \
                                            distance_between_pixel_and_sensor;
        const double time_travelled = (double)(distance_travelled/d_inputPointer_SoundSpeed[0]);
        
        // finding samples
        const int samples_travelled = (int)(time_travelled*d_inputPointer_SamplingFrequency[0]);

        // writing to the final output
        d_outputPointer[linearIndex2D_Output] = (double)samples_travelled;
    }
    
    // syncing threads
    __syncthreads();


    // Signal generation part 
    const int tidx_signal_generation = threadIdx.x + blockIdx.x*blockDim.x;
    const int tidy_signal_generation = threadIdx.y + blockIdx.y*blockDim.y;
    const int linearIndex2D_signal_generation = tidx_signal_generation + 
                                                tidy_signal_generation*gridDim.x;

    // Checking sensor validity
    if(linearIndex2D_signal_generation<d_inputPointer_NumSensors[0])
    {
        // updates
        const int current_sensor_loop = (int)linearIndex2D_signal_generation;
        const int num_samples_recording_time = \
            (int)(d_inputPointer_SamplingFrequency[0]*d_inputPointer_RecordingTime[0]);

        // Looping through all the pixels
        for(int i = 0; i<(int)d_inputDimensions_Image[0]; ++i)
        {
            for(int j = 0; j<(int)d_inputDimensions_Image[1]; ++j)
            {
                // finding linear index to the delay-num-samples
                const int linearIndex2D_loop_sample = \
                    i + \
                    j*d_inputDimensions_Image[0] + \
                    current_sensor_loop*d_inputDimensions_Image[0]*d_inputDimensions_Image[1];

                // obtaining num-sample delay
                const int num_samples_loop = d_outputPointer[linearIndex2D_loop_sample];

                // finding pixel-amplitude
                const int linearIndex2D_loop_pixel_amplitude = i + j*d_inputDimensions_Image[0];
                const int pixel_amplitude_loop = \
                    d_inputPointer_InputImage[linearIndex2D_loop_pixel_amplitude];

                // linear-indexing to the signal-matrix
                const int linearIndex2D_AmplitudeMatrix = \
                    num_samples_loop + \
                    current_sensor_loop*num_samples_recording_time;

                // accumulating
                d_outputPointer_AmplitudeMatrix[linearIndex2D_AmplitudeMatrix] = \
                    d_outputPointer_AmplitudeMatrix[linearIndex2D_AmplitudeMatrix] + \
                    (double)pixel_amplitude_loop;
            }
        }
    }

    // syncing threads
    __syncthreads();

}

__global__ void columnwiseConvolution(mwSize *d_outputDimensions_AmplitudeMatrix,
                                        double *d_outputPointer_AmplitudeMatrix,
                                        double *d_outputPointer_SignalMatrix,
                                        int numElements_TransmissionSignal,
                                        double *d_inputPointer_TransmissionSignal)
{
    // Declaring shared memory
    extern __shared__ double sharedMem[];
    
    // getting linear index into the pixels
    const int bidx = blockIdx.x; 
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;
    const int linearIndex2D_ForMainIndex = bidx + \
                                            bidy*gridDim.x + 
                                            bidz*gridDim.x*gridDim.y;

    // mapping from linear to index of base-index
    const int signal_x = linearIndex2D_ForMainIndex % (int)d_outputDimensions_AmplitudeMatrix[0];
    const int signal_y = (int)(linearIndex2D_ForMainIndex / (int)d_outputDimensions_AmplitudeMatrix[0]);

    // getting thread index
    const int tidx_kernel = threadIdx.x;
    const int tidy_kernel = threadIdx.y;
    const int tidz_kernel = threadIdx.z;
    const int linearIndex2D_kernel = tidx_kernel + \
                                        tidy_kernel*blockDim.x + \
                                        tidz_kernel*blockDim.x*blockDim.y;
    const int kernel_length = numElements_TransmissionSignal;

    // fetching kernel value
    bool meta_condition = \
        (signal_x < (int)d_outputDimensions_AmplitudeMatrix[0]) && \
        (signal_y < (int)d_outputDimensions_AmplitudeMatrix[1]) && \
        (linearIndex2D_kernel < kernel_length) && \
        ((signal_x - linearIndex2D_kernel) >= (int)0 ) && \
        ((signal_x - linearIndex2D_kernel) < (int)d_outputDimensions_AmplitudeMatrix[0]) \
        ;

    // Initializing kernel
    if(linearIndex2D_kernel<kernel_length) sharedMem[linearIndex2D_kernel] = (double)0;

    // Checking the validity before copying
    if(meta_condition)
    {
        // calculating dot product and copying to shared memory
        sharedMem[linearIndex2D_kernel] = \
            d_outputPointer_AmplitudeMatrix[linearIndex2D_ForMainIndex - linearIndex2D_kernel]* \
            d_inputPointer_TransmissionSignal[linearIndex2D_kernel];
    }
    __syncthreads();

    // Getting first thread of all the block to write
    if(linearIndex2D_kernel == 0 && (signal_x < (int)d_outputDimensions_AmplitudeMatrix[0]) && (signal_y < (int)d_outputDimensions_AmplitudeMatrix[1]) )
    {
        // accumulating shared-memory
        double accuSum = 0;
        for(int i = 0; i<min(numElements_TransmissionSignal, signal_x); ++i)
        {
            accuSum = accuSum + (double)sharedMem[i];
        }

        // copying to the results
        d_outputPointer_SignalMatrix[linearIndex2D_ForMainIndex] = \
            (double)accuSum;

    }

    // syncing threads
    __syncthreads();
}


// main function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // meta stuff
    cudaError_t err;

    // checking number of outputs
    if(nlhs!=3)
        mexErrMsgTxt("Number of Expected Outputs are Wrong  \n");

    // Fetching the Inputs 
    CustomGPUObject SensorArrayOrigin(prhs[0]); 
    CustomGPUObject SensorArrayDirection(prhs[1]); 
    CustomGPUObject SensorArraySpacing(prhs[2]); 
    CustomGPUObject NumSensors(prhs[3]); 
    CustomGPUObject EmitterLocation(prhs[4]); 
    CustomGPUObject InputImage(prhs[5]); 
    CustomGPUObject BottomLeftCoordinate(prhs[6]); 
    CustomGPUObject TopRightCoordinate(prhs[7]); 
    CustomGPUObject SoundSpeed(prhs[8]); 
    CustomGPUObject SignalFrequency(prhs[9]); 
    CustomGPUObject RecordingTime(prhs[10]); 
    CustomGPUObject SamplingFrequency(prhs[11]); 
    CustomGPUObject TransmissionSignal(prhs[12]); 

    // Sending data to GPU
    SensorArrayOrigin.copyFromHostToDevice();
    SensorArrayDirection.copyFromHostToDevice();
    SensorArraySpacing.copyFromHostToDevice();
    NumSensors.copyFromHostToDevice();
    EmitterLocation.copyFromHostToDevice();
    InputImage.copyFromHostToDevice();
    BottomLeftCoordinate.copyFromHostToDevice();
    TopRightCoordinate.copyFromHostToDevice();
    SoundSpeed.copyFromHostToDevice();
    SignalFrequency.copyFromHostToDevice();
    RecordingTime.copyFromHostToDevice();
    SamplingFrequency.copyFromHostToDevice();
    TransmissionSignal.copyFromHostToDevice();
    
    // setting up the output-dimensions
    mwSize *outputDimensions = (mwSize *)malloc(3*sizeof(mwSize));
    outputDimensions[0] = InputImage.inputDimensions[0]; 
    outputDimensions[1] = InputImage.inputDimensions[1]; 
    outputDimensions[2] = (mwSize)NumSensors.inputPointer_real[0];

    // creating output-matrix
    plhs[0] = mxCreateNumericArray((mwSize)3, outputDimensions, mxDOUBLE_CLASS, mxREAL);
    CustomGPUObject outputPointer_class(plhs[0]); 
    outputPointer_class.copyFromHostToDevice();
    
    // Creating Outputs for the Signal
    mwSize *outputDimensions_AmplitudeMatrix = (mwSize *)malloc(2*sizeof(mwSize));
    outputDimensions_AmplitudeMatrix[0] = (mwSize)(RecordingTime.inputPointer_real[0]*SamplingFrequency.inputPointer_real[0]);
    outputDimensions_AmplitudeMatrix[1] = (mwSize)NumSensors.inputPointer_real[0];

    // Creating Second Output
    plhs[1] = mxCreateNumericArray((mwSize)2, outputDimensions_AmplitudeMatrix, mxDOUBLE_CLASS, mxREAL);
    CustomGPUObject AmplitudeMatrix(plhs[1]); AmplitudeMatrix.copyFromHostToDevice();

    // Creating Third Output
    plhs[2] = mxCreateNumericArray(AmplitudeMatrix.numDimensions, AmplitudeMatrix.inputDimensions, mxDOUBLE_CLASS, mxREAL);
    CustomGPUObject SignalMatrix(plhs[2]); SignalMatrix.copyFromHostToDevice();

    // Calling the Function 
    dim3 blockConfiguration(32,32,1);
    dim3 gridConfiguration((int)InputImage.inputDimensions[0], (int)InputImage.inputDimensions[1], 1);
    PerPixel<<<gridConfiguration,
                blockConfiguration>>>(InputImage.d_inputDimensions,
                                        BottomLeftCoordinate.d_inputPointer_real,
                                        TopRightCoordinate.d_inputPointer_real,
                                        SensorArrayOrigin.d_inputPointer_real,
                                        SensorArrayDirection.d_inputPointer_real,
                                        outputPointer_class.d_inputPointer_real,
                                        NumSensors.d_inputPointer_real,
                                        SoundSpeed.d_inputPointer_real,
                                        EmitterLocation.d_inputPointer_real,
                                        SamplingFrequency.d_inputPointer_real,
                                        InputImage.d_inputPointer_real,
                                        RecordingTime.d_inputPointer_real,
                                        AmplitudeMatrix.d_inputPointer_real);


    // Prepping for function call
    const int grid_square_root_ceil = (int)sqrt(SignalMatrix.numElements) + 1;
    int test00 = (int)sqrt(TransmissionSignal.numElements) + 1;
    const int block_square_root_ceil = pow((((int)log2(test00)) + 1),2);
    dim3 blockConfiguration_convolution(block_square_root_ceil, block_square_root_ceil, 1);
    dim3 gridConfiguration_convolution((int)grid_square_root_ceil, (int)grid_square_root_ceil, 1);

    // Function Call
    columnwiseConvolution<<<gridConfiguration_convolution,
                            blockConfiguration_convolution,
                            TransmissionSignal.numElements*sizeof(double)>>>\
                            (AmplitudeMatrix.d_inputDimensions,
                            AmplitudeMatrix.d_inputPointer_real,
                            SignalMatrix.d_inputPointer_real,
                            TransmissionSignal.numElements,
                            TransmissionSignal.d_inputPointer_real);

    // Copying the data from Device to Host
    outputPointer_class.copyFromDeviceToHost();
    AmplitudeMatrix.copyFromDeviceToHost();
    SignalMatrix.copyFromDeviceToHost();

    // Device Shut down
    err = cudaDeviceSynchronize();
    err = cudaDeviceReset();
    
}
```