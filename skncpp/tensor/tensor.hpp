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
    std::vector<int> shape;
    float* data;
    size_t size;

    public:

    Tensor(const std::vector<int>& shape, Device device = Device::CPU);
    ~Tensor();

    const std::vector<int>& get_shape() const;
    float *get_data();
    void to(Device device);
    Tensor copy() const;
    void reshape(const std::vector<int>& new_shape);

    size_t index(const std::vector<int>& indices) const;
    float get(const std::vector<int>& indices) const;
    void set(const std::vector<int>& indices, float value);

    void zeros();
    void scalar_add(float scalar);
    void add(const Tensor& temp1, const Tensor& temp2);
    void multiply(const Tensor& temp1, const Tensor& temp2);

};

#endif