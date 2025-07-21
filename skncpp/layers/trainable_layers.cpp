/*
    Shubh Khandelwal
*/

#include <cmath>
#include "trainable_layers.hpp"

namespace skn
{

    TrainableLayer::TrainableLayer(Device device) : Layer(device)
    {}

    void TrainableLayer::to(Device device)
    {

        if (this->device == device)
        {
            return;
        }
        this->device = device;
        parameters[WEIGHTS].to(device);
        parameters[BIAS].to(device);
        gradients[WEIGHTS].to(device);
        gradients[BIAS].to(device);
        input.to(device);

    }

    const Tensor* TrainableLayer::get_parameters() const
    {
        return parameters;
    }

    const Tensor* TrainableLayer::get_gradients() const
    {
        return gradients;
    }

    Linear::Linear(int in_channels, int out_channels, Device device) : TrainableLayer(device), in_channels(in_channels), out_channels(out_channels)
    {

        parameters[WEIGHTS] = Tensor(1, in_channels, out_channels, device);
        parameters[WEIGHTS].zeros(); 
        parameters[BIAS] = Tensor(1, 1, out_channels, device);
        parameters[BIAS].zeros();

        gradients[WEIGHTS] = Tensor(1, in_channels, out_channels, device);
        gradients[BIAS] = Tensor(1, 1, out_channels, device);

    }

    Tensor Linear::forward(const Tensor& input)
    {
        this->input = input.copy();
        Tensor output(1, 1, out_channels, device);
        output.matrix_multiply(this->input, parameters[WEIGHTS]);
        output.add(output, parameters[BIAS]);
        return output;
    }

    Tensor Linear::backward(const Tensor& gradient)
    {
        input.reshape(1, in_channels, 1);
        gradients[WEIGHTS].matrix_multiply(input, gradient);
        gradients[BIAS] = gradient.copy();
        Tensor gradient_input(1, 1, in_channels, device);
        Tensor gradient_output = gradient.copy();
        gradient_output.reshape(1, out_channels, 1);
        gradient_input.matrix_multiply(parameters[WEIGHTS], gradient_output);
        gradient_input.reshape(1, 1, in_channels);
        return gradient_input;
    }

    Conv2D::Conv2D(int num_filters, int *kernel_size, int padding, int stride, Device device) : TrainableLayer(device), num_filters(num_filters), kernel_size(kernel_size), padding(padding), stride(stride)
    {

        parameters[WEIGHTS] = Tensor(num_filters, kernel_size[0], kernel_size[1], device);
        parameters[WEIGHTS].zeros(); 
        parameters[BIAS] = Tensor(num_filters, 1, 1, device);
        parameters[BIAS].zeros();

        gradients[WEIGHTS] = Tensor(num_filters, kernel_size[0], kernel_size[1], device);
        gradients[BIAS] = Tensor(num_filters, 1, 1, device);

    }

    Tensor Conv2D::forward(const Tensor& input)
    {
        this->input = input.copy();
        Tensor output(num_filters, floor((input.get_shape()[1] - kernel_size[0] + 2 * padding) / stride) + 1, floor((input.get_shape()[2] - kernel_size[1] + 2 * padding) / stride) + 1, device);
        output.convolve(input, parameters[WEIGHTS], padding, stride);
        output.scalar_add(output, parameters[BIAS]);
        return output;
    }

}