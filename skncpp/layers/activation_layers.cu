/*
    Shubh Khandelwal
*/

#include "activation_layers.hpp"
#include <iostream>

void sigmoid_CPU(float *input, float *output, int size)
{
    for (int i = 0; i < size; i++)
    {
        output[i] = 1 / (1 + exp(-input[i]));
    }
}

__global__ void sigmoid_GPU(float* input, float* output, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        output[i] = 1 / (1 + expf(-input[i]));
    }
}

void sigmoid1_CPU(float *input, float *output, int size)
{
    for (int i = 0; i < size; i++)
    {
        output[i] = exp(-input[i]) / ((1 + exp(-input[i])) * (1 + exp(-input[i])));
    }
}

__global__ void sigmoid1_GPU(float* input, float* output, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        output[i] = expf(-input[i]) / ((1 + expf(-input[i])) * (1 + expf(-input[i])));
    }
}

void tanh_CPU(float *input, float *output, int size)
{
    for (int i = 0; i < size; i++)
    {
        output[i] = (exp(input[i]) - exp(-input[i])) / (exp(input[i]) + exp(-input[i]));
    }
}

__global__ void tanh_GPU(float* input, float* output, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        output[i] = (expf(input[i]) - expf(-input[i])) / (expf(input[i]) + expf(-input[i]));
    }
}

void tanh1_CPU(float *input, float *output, int size)
{
    for (int i = 0; i < size; i++)
    {
        output[i] = 4 / ((exp(input[i]) + exp(-input[i])) * (exp(input[i]) + exp(-input[i])));
    }
}

__global__ void tanh1_GPU(float* input, float* output, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        output[i] = 4 / ((expf(input[i]) + expf(-input[i])) * (expf(input[i]) + expf(-input[i])));
    }
}

namespace skn
{

    ActivationLayer::ActivationLayer(Device device) : Layer(device)
    {}

    Tensor ActivationLayer::forward(const Tensor& input)
    {
        Tensor output = input.copy();
        if (device == Device::CPU)
        {
            output.function(activation_CPU);
        } else if (device == Device::GPU)
        {
            output.function(activation_GPU);
        } else
        {
            std::cerr << "UNKNOWN_DEVICE_ERROR" << std::endl;
            exit(1);
        }
        return output;
    }

    Tensor ActivationLayer::backward(const Tensor& gradient)
    {
        Tensor gradient_input = gradient.copy();
        if (device == Device::CPU)
        {
            gradient_input.function(activation1_CPU);
        } else if (device == Device::GPU)
        {
            gradient_input.function(activation1_GPU);
        } else
        {
            std::cerr << "UNKNOWN_DEVICE_ERROR" << std::endl;
            exit(1);
        }
        return gradient_input;
    }

    Sigmoid::Sigmoid(Device device) : ActivationLayer(device)
    {
        this->activation_CPU = sigmoid_CPU;
        this->activation_GPU = sigmoid_GPU;
        this->activation1_CPU = sigmoid1_CPU;
        this->activation1_GPU = sigmoid1_GPU;
    }

    Tanh::Tanh(Device device) : ActivationLayer(device)
    {
        this->activation_CPU = tanh_CPU;
        this->activation_GPU = tanh_GPU;
        this->activation1_CPU = tanh1_CPU;
        this->activation1_GPU = tanh1_GPU;
    }

}