#include "op.h"
#include "utils.h"

void ffn_forward(cublasHandle_t handle, int batch_size, int input_features,
                 int hidden_features, int output_features, const float *d_input,
                 const float *d_W1, const float *d_b1, const float *d_W2,
                 const float *d_b2, float *d_intermediate_buffer,
                 float *d_output) {

  const float alpha = 1.0f;
  const float beta = 0.0f;

  CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_features,
                           batch_size, input_features, &alpha, d_W1,
                           hidden_features, d_input, input_features, &beta,
                           d_intermediate_buffer, hidden_features));

  dim3 threads_per_block(16, 16);
  dim3 num_blocks((hidden_features + threads_per_block.x - 1) /
                      threads_per_block.x,
                  (batch_size + threads_per_block.y - 1) / threads_per_block.y);
  add_bias_and_relu_kernel<<<num_blocks, threads_per_block>>>(
      d_intermediate_buffer, d_b1, batch_size, hidden_features);
  CHECK_CUDA(cudaGetLastError()); 

  CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, output_features,
                           batch_size, hidden_features, &alpha, d_W2,
                           output_features, d_intermediate_buffer,
                           hidden_features, &beta, d_output, output_features));

  dim3 threads_per_block2(16, 16);
  dim3 num_blocks2(
      (output_features + threads_per_block2.x - 1) / threads_per_block2.x,
      (batch_size + threads_per_block2.y - 1) / threads_per_block2.y);
  add_bias_kernel<<<num_blocks2, threads_per_block2>>>(
      d_output, d_b2, batch_size, output_features);
  CHECK_CUDA(cudaGetLastError()); 
}