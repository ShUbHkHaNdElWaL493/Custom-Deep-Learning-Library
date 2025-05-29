/*
    Shubh Khandelwal
*/

#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <string>
#include <tensor/tensor.hpp>
#include <vector>

class Layer
{

    public:

    virtual ~Layer() = default;

    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;

};

class TrainableLayer : public Layer
{

    protected:

    enum
    {
        WEIGHTS = 0,
        BIAS = 1
    };

    int in_channels, out_channels;
    Tensor parameters[2], gradients[2], input;

    public:

    TrainableLayer(int in_channels, int out_channels) : in_channels(in_channels), out_channels(out_channels)
    {}

    const Tensor* get_parameters() const
    {
        return parameters;
    }

    const Tensor* get_gradients() const
    {
        return gradients;
    }

};

class Linear : public TrainableLayer
{

    public:

    Linear(int in_channels, int out_channels, Device device) : TrainableLayer(in_channels, out_channels)
    {

        gradients[WEIGHTS] = Tensor(1, out_channels, in_channels, device);
        gradients[BIAS] = Tensor(1, 1, out_channels, device);

        parameters[WEIGHTS] = Tensor(1, in_channels, out_channels, device);
        parameters[BIAS] = Tensor(1, 1, out_channels, device);

        parameters[WEIGHTS].zeros(); 
        parameters[BIAS].zeros();

    }

    Tensor forward(const Tensor& input) override
    {
        Tensor output(1, 1, out_channels, input.get_device());
        output.matrix_multiply(input, parameters[WEIGHTS]);
        output.add(output, parameters[BIAS]);
        return output;
    }

    Tensor backward(const Tensor& gradient) override
    {
        const int *shape = input.get_shape();
        Tensor gradient_input(shape[0], shape[1], shape[2], input.get_device());
        return gradient_input;
    }

};

#endif