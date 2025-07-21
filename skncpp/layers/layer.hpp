/*
    Shubh Khandelwal
*/

#ifndef LAYER_HPP
#define LAYER_HPP

#include <tensor/tensor.hpp>

namespace skn
{
    class Layer
    {

        protected:

        Device device;

        public:

        Layer(Device device);
        virtual ~Layer() = default;

        virtual void to(Device device);
        virtual Tensor forward(const Tensor& input) = 0;
        virtual Tensor backward(const Tensor& gradient) = 0;

    };
}

#endif