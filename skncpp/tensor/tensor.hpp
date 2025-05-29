/*
    Shubh Khandelwal
*/

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cassert>
#include <vector>

#define CUDA_CHECK(cmd) { \
        cudaError_t error = cmd; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    }

enum class Device
{
    CPU,
    GPU
};

class Tensor
{

    private:

    Device device;
    int shape[3];
    float* data;
    size_t size;

    public:

    Tensor();
    Tensor(int channels, int rows, int columns, Device device = Device::CPU);
    ~Tensor();

    const Device get_device() const;
    const int* get_shape() const;
    void to(Device device);
    Tensor copy() const;
    void reshape(int channels, int rows, int columns);

    size_t index(int channels, int row, int column) const;
    float get(int channels, int row, int column) const;
    void set(int channels, int row, int column, float value);

    void zeros();
    void scalar_add(float scalar);
    void add(const Tensor& temp1, const Tensor& temp2);
    void multiply(const Tensor& temp1, const Tensor& temp2);
    void matrix_multiply(const Tensor& temp1, const Tensor& temp2);

};

#endif