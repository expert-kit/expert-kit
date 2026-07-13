#pragma once

#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

void ffn_forward(cublasHandle_t handle, int batch_size, int input_features,
                 int hidden_features, int output_features, const float *d_input,
                 const float *d_W1, const float *d_b1, const float *d_W2,
                 const float *d_b2, float *d_intermediate_buffer,
                 float *d_output);

__global__ void add_bias_and_relu_kernel(float *matrix, const float *bias,
                                         int rows, int cols);

__global__ void add_bias_kernel(float *matrix, const float *bias, int rows,
                                int cols);

__global__ void expert_server(struct doca_gpu_dev_rdma *rdma_gpu,
                              struct doca_gpu_buf_arr *buf_arr_hidden);