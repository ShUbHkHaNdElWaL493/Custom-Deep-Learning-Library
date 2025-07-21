/*
    Shubh Khandelwal
*/

#ifndef ACTIVATION_LAYERS_HPP
#define ACTIVATION_LAYERS_HPP

#include <layers/layer.hpp>

namespace skn
{

    class ActivationLayer : public Layer
    {

        protected:

        void (*activation_CPU)(float* input, float* output, int size);
        void (*activation_GPU)(float* input, float* output, int size);
        void (*activation1_CPU)(float* input, float* output, int size);
        void (*activation1_GPU)(float* input, float* output, int size);

        public:

        ActivationLayer(Device device);
        virtual ~ActivationLayer() = default;

        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& gradient) override;

    };

    class Sigmoid : public ActivationLayer
    {

        public:

        Sigmoid(Device device);

    };

}

#endif