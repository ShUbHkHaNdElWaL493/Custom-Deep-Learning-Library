/*
    Shubh Khandelwal
*/

#ifndef TRAINABLE_LAYERS_HPP
#define TRAINABLE_LAYERS_HPP

#include <layers/layer.hpp>

namespace skn
{

    class TrainableLayer : public Layer
    {

        protected:

        enum
        {
            WEIGHTS = 0,
            BIAS = 1
        };

        Tensor parameters[2], gradients[2], input;

        public:

        TrainableLayer(Device device);
        virtual ~TrainableLayer() = default;

        void to(Device device) override;
        const Tensor* get_parameters() const;
        const Tensor* get_gradients() const;

    };

    class Linear : public TrainableLayer
    {

        private:

        int in_channels, out_channels;

        public:

        Linear(int in_channels, int out_channels, Device device);

        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& gradient) override;

    };

    class Conv2D : public TrainableLayer
    {

        private:
        
        int num_filters, padding, stride;
        int *kernel_size;

        public:

        Conv2D(int num_filters, int *kernel_size, int padding = 0, int stride = 1, Device device);

        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& gradient) override;

    };

}

#endif