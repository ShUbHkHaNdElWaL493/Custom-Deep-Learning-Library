/*
    Shubh Khandelwal
*/

#include <cuda_runtime.h>
#include <iostream>
#include "tensor.hpp"

__global__ void kernel_add(const float* a, const float* b, float* out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        out[i] = a[i] + b[i];
    }
}

__global__ void kernel_multiply(const float* a, const float* b, float* out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        out[i] = a[i] * b[i];
    }
}

__global__ void kernel_scalar_add(float* a, float scalar, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        a[i] += scalar;
    }
}

Tensor::Tensor(const std::vector<int>& shape, Device device) : shape(shape), device(device)
{

    size = 1;
    for (int dim : shape) size *= dim;

    if (device == Device::CPU)
    {
        data = new float[size];
    } else if (device == Device::GPU)
    {
        CUDA_CHECK(cudaMalloc(&data, size * sizeof(float)));
    } else
    {
        std::cerr << "UNKNOWN_DEVICE_ERROR" << std::endl;
        exit(1);
    }

}

Tensor::~Tensor()
{
    if (device == Device::CPU)
    {
        delete[] data;
    } else if (device == Device::GPU)
    {
        CUDA_CHECK(cudaFree(data));
    } else
    {
        std::cerr << "UNKNOWN_DEVICE_ERROR" << std::endl;
        exit(1);
    }
}

const std::vector<int>& Tensor::get_shape() const
{
    return shape;
}

float* Tensor::get_data()
{
    return data;
}

void Tensor::to(Device device)
{

    if (this->device == device)
    {
        return;
    }

    if (device == Device::CPU)
    {
        float *temp = new float[size];
        CUDA_CHECK(cudaMemcpy((void*) temp, (void*) data, size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(data));
        data = temp;
    } else if (device == Device::GPU)
    {
        float *temp;
        CUDA_CHECK(cudaMalloc(&temp, size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy((void*) temp, (void*) data, size * sizeof(float), cudaMemcpyHostToDevice));
        delete[] data;
        data = temp;
    } else
    {
        std::cerr << "UNKNOWN_DEVICE_ERROR" << std::endl;
        exit(1);
    }

    this->device = device;

}

Tensor Tensor::copy() const
{
    Tensor clone(shape, device);
    if (device == Device::CPU)
    {
        std::copy(data, data + size, clone.data);
    } else if (device == Device::GPU)
    {
        CUDA_CHECK(cudaMemcpy(clone.data, data, size * sizeof(float), cudaMemcpyDeviceToDevice));
    } else
    {
        std::cerr << "UNKNOWN_DEVICE_ERROR" << std::endl;
        exit(1);
    }
    return clone;
}

void Tensor::reshape(const std::vector<int>& new_shape)
{
    size_t new_size = 1;
    for (int dim : new_shape)
    {
        new_size *= dim;
    }
    if (new_size != size)
    {
        std::cerr << "SIZE_MISMATCH_ERROR" << std::endl;
        exit(2);
    }
    shape = new_shape;
}

size_t Tensor::index(const std::vector<int>& indices) const
{

    if (indices.size() != shape.size())
    {
        std::cerr << "DIMENSION_MISMATCH_ERROR" << std::endl;
        exit(3);
    }

    size_t idx = 0;
    size_t multiplier = 1;
    for (int i = shape.size() - 1; i >= 0; i--)
    {
        idx += indices[i] * multiplier;
        multiplier *= shape[i];
    }

    return idx;

}

float Tensor::get(const std::vector<int>& indices) const
{
    if (device != Device::CPU)
    {
        std::cerr << "ILLEGAL_DEVICE_ERROR" << std::endl;
        exit(4);
    }
    return data[index(indices)];
}

void Tensor::set(const std::vector<int>& indices, float value)
{
    if (device != Device::CPU)
    {
        std::cerr << "ILLEGAL_DEVICE_ERROR" << std::endl;
        exit(4);
    }
    data[index(indices)] = value;
}

void Tensor::zeros()
{
    if (device == Device::CPU)
    {
        std::fill(data, data + size, 0.0f);
    } else if (device == Device::GPU)
    {
        CUDA_CHECK(cudaMemset(data, 0, size * sizeof(float)));
    } else
    {
        std::cerr << "UNKNOWN_DEVICE_ERROR" << std::endl;
        exit(1);
    }
}

void Tensor::scalar_add(float scalar)
{
    if (device == Device::CPU)
    {
        for (size_t i = 0; i < size; i++)
        {
            data[i] += scalar;
        }
    } else if (device == Device::GPU)
    {
        int threads = 32;
        int blocks = (size + threads - 1) / threads;
        kernel_scalar_add<<<blocks, threads>>>(data, scalar, size);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else
    {
        std::cerr << "UNKNOWN_DEVICE_ERROR" << std::endl;
        exit(1);
    }
}

void Tensor::add(const Tensor& temp1, const Tensor& temp2)
{

    if (shape != temp1.shape || shape != temp2.shape || device != temp1.device || device != temp2.device)
    {
        std::cerr << "SHAPE_OR_DEVICE_MISMATCH_ERROR" << std::endl;
        exit(5);
    }

    if (device == Device::CPU)
    {
        for (size_t i = 0; i < size; i++)
        {
            data[i] = temp1.data[i] + temp2.data[i];
        }
    } else if (device == Device::GPU)
    {
        int threads = 32;
        int blocks = (size + threads - 1) / threads;
        kernel_add<<<blocks, threads>>>(temp1.data, temp2.data, data, size);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else
    {
        std::cerr << "UNKNOWN_DEVICE_ERROR" << std::endl;
        exit(1);
    }
    
}

void Tensor::multiply(const Tensor& temp1, const Tensor& temp2)
{

    if (shape != temp1.shape || shape != temp2.shape || device != temp1.device || device != temp2.device)
    {
        std::cerr << "SHAPE_OR_DEVICE_MISMATCH_ERROR" << std::endl;
        exit(5);
    }

    if (device == Device::CPU)
    {
        for (size_t i = 0; i < size; i++)
        {
            data[i] = temp1.data[i] * temp2.data[i];
        }
    } else if (device == Device::GPU)
    {
        int threads = 32;
        int blocks = (size + threads - 1) / threads;
        kernel_multiply<<<blocks, threads>>>(temp1.data, temp2.data, data, size);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else
    {
        std::cerr << "UNKNOWN_DEVICE_ERROR" << std::endl;
        exit(1);
    }

}