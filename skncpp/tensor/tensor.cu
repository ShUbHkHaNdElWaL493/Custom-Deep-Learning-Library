/*
    Shubh Khandelwal
*/

#include <cuda_runtime.h>
#include <iostream>
#include "tensor.hpp"

__global__ void kernel_scalar_add(const float* A, const float* B, float* out, int C, int I, int J)
{
    
    int c = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < I && j < J)
    {
        int idxA = (c * I + i) * J + j;
        int idxB = c;
        int idxOut = (c * I + i) * J + j;
        out[idxOut] = A[idxA] + B[idxB];
    }

}

__global__ void kernel_add(const float* A, const float* B, float* out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        out[i] = A[i] + B[i];
    }
}

__global__ void kernel_multiply(const float* A, const float* B, float* out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        out[i] = A[i] * B[i];
    }
}

__global__ void kernel_matrix_multiply(const float* A, const float* B, float* out, int C, int I, int K, int J)
{

    int c = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < I && j < J)
    {
        float sum = 0;
        for (int k = 0; k < K; ++k)
        {
            int idxA = (c * I + i) * K + k;
            int idxB = (c * K + k) * J + j;
            sum += A[idxA] * B[idxB];
        }
        int idxOut = (c * I + i) * J + j;
        out[idxOut] = sum;
    }

}

__global__ void kernel_convolve(const float* A, const float* B, float* out, int C, int I, int J, int K, int L, int M, int N, int stride, int padding)
{

    int c = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < I && j < J)
    {
        float sum = 0;
        for (int k = 0; k < K; k++)
        {
            for (int l = 0; l < L; l++)
            {
                int m = i * stride + k - padding;
                int n = j * stride + l - padding;
                if (m >= 0 && n >= 0 && m < M && n < N)
                {
                    int idxA = (c * M + m) * N + n;
                    int idxB = (c * K + k) * L + l;
                    sum += A[idxA] * B[idxB];
                }
            }
        }
        int idxOut = (c * I + i) * J + j;
        out[idxOut] = sum;
    }

}

namespace skn
{

    Tensor::Tensor()
    {}

    Tensor::Tensor(int channels, int rows, int columns, Device device) : device(device)
    {

        shape[0] = channels;
        shape[1] = rows;
        shape[2] = columns;

        size = 1;
        for (int dim : shape)
        {
            size *= dim;
        }

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

    Device Tensor::get_device() const
    {
        return device;
    }

    const int* Tensor::get_shape() const
    {
        return shape;
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
        Tensor clone(shape[0], shape[1], shape[2], device);
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

    void Tensor::reshape(int channels, int rows, int columns)
    {
        size_t new_size = channels * rows * columns;
        if (new_size != size)
        {
            std::cerr << "DIMENSION_MISMATCH_ERROR" << std::endl;
            exit(4);
        }
        shape[0] = channels;
        shape[1] = rows;
        shape[2] = columns;
    }

    float Tensor::get(int channel, int row, int column) const
    {

        if (device != Device::CPU)
        {
            std::cerr << "ILLEGAL_DEVICE_ERROR" << std::endl;
            exit(2);
        }

        if ((channel >= shape[0]) || (row >= shape[1]) || (column >= shape[2]))
        {
            std::cerr << "DIMENSION_MISMATCH_ERROR" << std::endl;
            exit(4);
        }

        return data[(channel * shape[1] + row) * shape[2] + column];

    }

    void Tensor::set(int channel, int row, int column, float value)
    {

        if (device != Device::CPU)
        {
            std::cerr << "ILLEGAL_DEVICE_ERROR" << std::endl;
            exit(2);
        }

        if ((channel >= shape[0]) || (row >= shape[1]) || (column >= shape[2]))
        {
            std::cerr << "DIMENSION_MISMATCH_ERROR" << std::endl;
            exit(4);
        }

        data[(channel * shape[1] + row) * shape[2] + column] = value;

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

    void Tensor::scalar_add(const Tensor& temp1, const Tensor& temp2)
    {

        if ((device != temp1.device) || (device != temp2.device))
        {
            std::cerr << "DEVICE_MISMATCH_ERROR" << std::endl;
            exit(3);
        }

        if ((shape[0] != temp1.shape[0]) || (shape[0] != temp2.shape[0]) \
        || (shape[1] != temp1.shape[1]) || (temp2.shape[1] != 1) \
        || (shape[2] != temp1.shape[2]) || (temp2.shape[2] != 1))
        {
            std::cerr << "DIMENSION_MISMATCH_ERROR" << std::endl;
            exit(4);
        }

        int C = shape[0];
        int I = shape[1];
        int J = shape[2];

        if (device == Device::CPU)
        {
            for (int c = 0; c < C; c++)
            {
                for (int i = 0; i < I; i++)
                {
                    for (int j = 0; j < J; j++)
                    {
                        set(c, i, j, (temp1.get(c, i, j) + temp2.get(c, 0, 0)));
                    }
                }
            }
        } else if (device == Device::GPU)
        {
            dim3 blockDim(16, 16);
            dim3 gridDim((J + blockDim.x - 1) / blockDim.x, (I + blockDim.y - 1) / blockDim.y, C);
            kernel_scalar_add<<<gridDim, blockDim>>>(temp1.data, temp2.data, data, C, I, J);
            CUDA_CHECK(cudaDeviceSynchronize());
        } else
        {
            std::cerr << "UNKNOWN_DEVICE_ERROR" << std::endl;
            exit(1);
        }

    }

    void Tensor::add(const Tensor& temp1, const Tensor& temp2)
    {

        if ((device != temp1.device) || (device != temp2.device))
        {
            std::cerr << "DEVICE_MISMATCH_ERROR" << std::endl;
            exit(3);
        }

        if ((shape[0] != temp1.shape[0]) || (shape[0] != temp2.shape[0]) \
        || (shape[1] != temp1.shape[1]) || (shape[1] != temp2.shape[1]) \
        || (shape[2] != temp1.shape[2]) || (shape[2] != temp2.shape[2]))
        {
            std::cerr << "DIMENSION_MISMATCH_ERROR" << std::endl;
            exit(4);
        }

        if (device == Device::CPU)
        {
            for (size_t i = 0; i < size; i++)
            {
                data[i] = temp1.data[i] + temp2.data[i];
            }
        } else if (device == Device::GPU)
        {
            int threads = 256;
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

        if ((device != temp1.device) || (device != temp2.device))
        {
            std::cerr << "DEVICE_MISMATCH_ERROR" << std::endl;
            exit(3);
        }

        if ((shape[0] != temp1.shape[0]) || (shape[0] != temp2.shape[0]) \
        || (shape[1] != temp1.shape[1]) || (shape[1] != temp2.shape[1]) \
        || (shape[2] != temp1.shape[2]) || (shape[2] != temp2.shape[2]))
        {
            std::cerr << "DIMENSION_MISMATCH_ERROR" << std::endl;
            exit(4);
        }

        if (device == Device::CPU)
        {
            for (size_t i = 0; i < size; i++)
            {
                data[i] = temp1.data[i] * temp2.data[i];
            }
        } else if (device == Device::GPU)
        {
            int threads = 256;
            int blocks = (size + threads - 1) / threads;
            kernel_multiply<<<blocks, threads>>>(temp1.data, temp2.data, data, size);
            CUDA_CHECK(cudaDeviceSynchronize());
        } else
        {
            std::cerr << "UNKNOWN_DEVICE_ERROR" << std::endl;
            exit(1);
        }

    }

    void Tensor::matrix_multiply(const Tensor& temp1, const Tensor& temp2)
    {

        if ((device != temp1.device) || (device != temp2.device))
        {
            std::cerr << "DEVICE_MISMATCH_ERROR" << std::endl;
            exit(3);
        }

        if ((shape[0] != temp1.shape[0]) \
        || (shape[0] != temp2.shape[0]) \
        || (shape[1] != temp1.shape[1]) \
        || (shape[2] != temp2.shape[2]) \
        || (temp1.shape[2] != temp2.shape[1]))
        {
            std::cerr << "DIMENSION_MISMATCH_ERROR" << std::endl;
            exit(4);
        }

        int C = shape[0];
        int I = shape[1];
        int J = shape[2];
        int K = temp1.shape[2];

        if (device == Device::CPU)
        {
            for (int c = 0; c < C; c++)
            {
                for (int i = 0; i < I; i++)
                {
                    for (int j = 0; j < J; j++)
                    {
                        float sum = 0;
                        for (int k = 0; k < K; k++)
                        {
                            sum += temp1.get(c, i, k) * temp2.get(c, k, j);
                        }
                        set(c, i, j, sum);
                    }
                }
            }
        } else if (device == Device::GPU)
        {
            dim3 blockDim(16, 16);
            dim3 gridDim((J + blockDim.x - 1) / blockDim.x, (I + blockDim.y - 1) / blockDim.y, C);
            kernel_matrix_multiply<<<gridDim, blockDim>>>(temp1.data, temp2.data, data, C, I, K, J);
            CUDA_CHECK(cudaDeviceSynchronize());
        } else
        {
            std::cerr << "UNKNOWN_DEVICE_ERROR" << std::endl;
            exit(1);
        }

    }

    void Tensor::convolve(const Tensor &temp1, const Tensor &temp2, int padding, int stride)
    {

        if ((device != temp1.device) || (device != temp2.device))
        {
            std::cerr << "DEVICE_MISMATCH_ERROR" << std::endl;
            exit(3);
        }

        if ((shape[0] != temp1.shape[0]) \
        || (shape[0] != temp2.shape[0]) \
        || (shape[1] != int(floor((temp1.shape[1] - temp2.shape[1] + 2 * padding) / stride + 1))) \
        || (shape[2] != int(floor((temp1.shape[2] - temp2.shape[2] + 2 * padding) / stride + 1))))
        {
            std::cerr << "DIMENSION_MISMATCH_ERROR" << std::endl;
            exit(4);
        }

        int C = shape[0];
        int I = shape[1];
        int J = shape[2];
        int K = temp2.shape[1];
        int L = temp2.shape[2];
        int M = temp1.shape[1];
        int N = temp1.shape[2];

        if (device == Device::CPU)
        {
            for (int c = 0; c < C; c++)
            {
                for (int i = 0; i < I; i++)
                {
                    for (int j = 0; j < J; j++)
                    {
                        float sum = 0;
                        for (int k = 0; k < K; k++)
                        {
                            for (int l = 0; l < L; l++)
                            {
                                int m = i * stride + k - padding;
                                int n = j * stride + l - padding;
                                if (m >= 0 && n >= 0 && m < M && n < N)
                                {
                                    sum += temp1.get(c, m, n) * temp2.get(c, k, l);
                                }
                            }
                        }
                        set(c, i, j, sum);
                    }
                }
            }
        } else if (device == Device::GPU)
        {
            dim3 blockDim(16, 16);
            dim3 gridDim((J + blockDim.x - 1) / blockDim.x, (I + blockDim.y - 1) / blockDim.y, C);
            kernel_convolve<<<gridDim, blockDim>>>(temp1.data, temp2.data, data, C, I, J, K, L, M, N, stride, padding);
            CUDA_CHECK(cudaDeviceSynchronize());
        } else
        {
            std::cerr << "UNKNOWN_DEVICE_ERROR" << std::endl;
            exit(1);
        }

    }

    void Tensor::function(void (*activation)(float* input, float* output, int size))
    {
        if (device == Device::CPU)
        {
            activation(data, data, size);
        } else if (device == Device::GPU)
        {
            int threads = 256;
            int blocks = (size + threads - 1) / threads;
            activation<<<blocks, threads>>>(data, data, size);
            CUDA_CHECK(cudaDeviceSynchronize());
        } else
        {
            std::cerr << "UNKNOWN_DEVICE_ERROR" << std::endl;
            exit(1);
        }
    }

}