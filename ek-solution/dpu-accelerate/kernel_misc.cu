#include "op.h"

__global__ void add_bias_and_relu_kernel(float* matrix, const float* bias, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        // Add bias and apply ReLU activation function: f(x) = max(0, x)
        matrix[index] = fmaxf(0.0f, matrix[index] + bias[col]);
    }
}
__global__ void add_bias_kernel(float* matrix, const float* bias, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        matrix[index] = matrix[index] + bias[col];
    }
}