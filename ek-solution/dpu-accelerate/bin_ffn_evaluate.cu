
#include "op.h"
#include "utils.h"
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>


// Function to initialize a matrix with random values
void initialize_matrix(float *mat, int rows, int cols) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  for (int i = 0; i < rows * cols; ++i) {
    mat[i] = dis(gen);
  }
}

// Function to perform FFN on CPU for verification
void ffn_cpu(int batch_size, int input_features, int hidden_features,
             int output_features, const float *h_input, const float *h_W1,
             const float *h_b1, const float *h_W2, const float *h_b2,
             float *h_output_cpu) {

  std::vector<float> h_intermediate(batch_size * hidden_features);

  // First linear layer: Z1 = Input @ W1
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < hidden_features; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < input_features; ++k) {
        sum += h_input[i * input_features + k] * h_W1[k * hidden_features + j];
      }
      // Add bias and apply ReLU
      h_intermediate[i * hidden_features + j] = fmaxf(0.0f, sum + h_b1[j]);
    }
  }

  // Second linear layer: Output = A1 @ W2
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < output_features; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < hidden_features; ++k) {
        sum += h_intermediate[i * hidden_features + k] *
               h_W2[k * output_features + j];
      }
      // Add final bias
      h_output_cpu[i * output_features + j] = sum + h_b2[j];
    }
  }
}

// Function to compare CPU and GPU results
void verify_results(int size, const float *h_output_gpu,
                    const float *h_output_cpu) {
  double total_error = 0.0;
  for (int i = 0; i < size; ++i) {
    total_error += std::abs(h_output_gpu[i] - h_output_cpu[i]);
  }
  double avg_error = total_error / size;
  std::cout << "Average difference between GPU and CPU results: " << avg_error
            << std::endl;
  if (avg_error < 1e-5) {
    std::cout << "Verification PASSED!" << std::endl;
  } else {
    std::cout << "Verification FAILED!" << std::endl;
  }

  // Print a few examples
  std::cout << "\n--- Sample Results (GPU vs CPU) ---" << std::endl;
  std::cout << std::fixed << std::setprecision(6);
  for (int i = 0; i < std::min(10, size); ++i) {
    std::cout << "Index " << std::setw(3) << i << ": GPU=" << std::setw(10)
              << h_output_gpu[i] << ", CPU=" << std::setw(10) << h_output_cpu[i]
              << std::endl;
  }
}

// --- Main ---
int main() {
  // --- Configuration ---
  const int batch_size = 128;
  const int input_features = 768;
  const int hidden_features = 3072; // Typically 4 * input_features
  const int output_features = 768;  // Typically same as input_features

  std::cout << "--- FFN Operator Test ---" << std::endl;
  std::cout << "Batch Size: " << batch_size << std::endl;
  std::cout << "Input Features: " << input_features << std::endl;
  std::cout << "Hidden Features: " << hidden_features << std::endl;
  std::cout << "Output Features: " << output_features << std::endl;
  std::cout << "-------------------------" << std::endl;

  // --- Host Memory Allocation and Initialization ---
  std::cout << "Allocating and initializing host memory..." << std::endl;
  std::vector<float> h_input(batch_size * input_features);
  std::vector<float> h_W1(input_features * hidden_features);
  std::vector<float> h_b1(hidden_features);
  std::vector<float> h_W2(hidden_features * output_features);
  std::vector<float> h_b2(output_features);
  std::vector<float> h_output_gpu(batch_size * output_features);
  std::vector<float> h_output_cpu(batch_size * output_features);

  initialize_matrix(h_input.data(), batch_size, input_features);
  initialize_matrix(h_W1.data(), input_features, hidden_features);
  initialize_matrix(h_b1.data(), 1, hidden_features);
  initialize_matrix(h_W2.data(), hidden_features, output_features);
  initialize_matrix(h_b2.data(), 1, output_features);

  // --- Device Memory Allocation ---
  std::cout << "Allocating device memory..." << std::endl;
  float *d_input, *d_W1, *d_b1, *d_W2, *d_b2, *d_intermediate_buffer, *d_output;
  CHECK_CUDA(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_W1, h_W1.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b1, h_b1.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_W2, h_W2.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b2, h_b2.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_intermediate_buffer,
                        batch_size * hidden_features * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_output, h_output_gpu.size() * sizeof(float)));

  // --- Copy Data from Host to Device ---
  std::cout << "Copying data from host to device..." << std::endl;
  CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_W1, h_W1.data(), h_W1.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b1, h_b1.data(), h_b1.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_W2, h_W2.data(), h_W2.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b2, h_b2.data(), h_b2.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // --- Setup cuBLAS and Execute FFN ---
  std::cout << "Executing FFN forward pass on GPU..." << std::endl;
  cublasHandle_t cublas_handle;
  CHECK_CUBLAS(cublasCreate(&cublas_handle));

  ffn_forward(cublas_handle, batch_size, input_features, hidden_features,
              output_features, d_input, d_W1, d_b1, d_W2, d_b2,
              d_intermediate_buffer, d_output);

  CHECK_CUDA(
      cudaDeviceSynchronize()); // Wait for all GPU operations to complete

  // --- Copy Result from Device to Host ---
  std::cout << "Copying result from device to host..." << std::endl;
  CHECK_CUDA(cudaMemcpy(h_output_gpu.data(), d_output,
                        h_output_gpu.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // --- Verification ---
  std::cout << "Executing FFN on CPU for verification..." << std::endl;
  ffn_cpu(batch_size, input_features, hidden_features, output_features,
          h_input.data(), h_W1.data(), h_b1.data(), h_W2.data(), h_b2.data(),
          h_output_cpu.data());

  std::cout << "\n--- Verification ---" << std::endl;
  verify_results(h_output_gpu.size(), h_output_gpu.data(), h_output_cpu.data());

  // --- Cleanup ---
  std::cout << "\nCleaning up..." << std::endl;
  CHECK_CUBLAS(cublasDestroy(cublas_handle));
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_W1));
  CHECK_CUDA(cudaFree(d_b1));
  CHECK_CUDA(cudaFree(d_W2));
  CHECK_CUDA(cudaFree(d_b2));
  CHECK_CUDA(cudaFree(d_intermediate_buffer));
  CHECK_CUDA(cudaFree(d_output));

  std::cout << "Done." << std::endl;
  return 0;
}