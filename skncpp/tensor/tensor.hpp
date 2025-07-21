/*
    Shubh Khandelwal
*/

#ifndef TENSOR_HPP
#define TENSOR_HPP

#define CUDA_CHECK(cmd) { \
    cudaError_t error = cmd; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

namespace skn
{

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

        Device get_device() const;
        const int* get_shape() const;
        void to(Device device);
        Tensor copy() const;
        void reshape(int channels, int rows, int columns);

        float get(int channels, int row, int column) const;
        void set(int channels, int row, int column, float value);

        void zeros();
        void scalar_add(const Tensor& temp1, const Tensor& temp2);
        void add(const Tensor& temp1, const Tensor& temp2);
        void multiply(const Tensor& temp1, const Tensor& temp2);
        void matrix_multiply(const Tensor& temp1, const Tensor& temp2);
        void convolve(const Tensor& temp1, const Tensor& temp2, int padding = 0, int stride = 1);
        void function(void (*activation)(float* input, float* output, int size));

    };

}

#endif