/*
    Shubh Khandelwal
*/

#include "layer.hpp"

namespace skn
{

    Layer::Layer(Device device) : device(device)
    {}

    void Layer::to(Device device)
    {
        if (this->device == device)
        {
            return;
        }
        this->device = device;
    }
    
}