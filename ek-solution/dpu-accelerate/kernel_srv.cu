#include "constant.h"
__global__ void expert_server(struct doca_gpu_dev_rdma *rdma_gpu,
                              struct doca_gpu_buf_arr *buf_arr_hidden,
                              struct doca_gpu_buf_arr *buf_arr_flag,
                              uint32_t connection_index) {
  doca_error_t result;
  struct doca_gpu_buf *lbuf_hidden;
  struct doca_gpu_buf *rbuf_flag;

  // enum doca_rdma_opcode opcode;
  int buf_index = 0;
  uint32_t num_ops;
  struct doca_gpu_dev_rdma_r *rdma_gpu_r;
  uint32_t imm_val[2];
  uint32_t conn_idx[2];

  result = doca_gpu_dev_rdma_get_recv(rdma_gpu, &rdma_gpu_r);
  if (result != DOCA_SUCCESS)
    printf("Error %d doca_gpu_dev_buf_get_buf\n", result);

  doca_gpu_dev_buf_get_buf(buf_arr_hidden, buf_index, &lbuf_hidden);

  // receive hidden vector from client
  result = doca_gpu_dev_rdma_recv_strong(rdma_gpu_r, lbuf_hidden, HIDDEN_SIZE,
                                         (threadIdx.x * HIDDEN_SIZE), 0);
  if (result != DOCA_SUCCESS)
    printf("Error %d doca_gpu_dev_rdma_recv_strong \n", result);

  result = doca_gpu_dev_rdma_write_inline_strong(
      rdma_gpu, 0, rbuf_flag, buf_index, 1, sizeof(uint8_t), 0,
      DOCA_GPU_RDMA_WRITE_FLAG_NONE);

  if (result != DOCA_SUCCESS)
    printf("Error doca_gpu_dev_rdma_write_inline_strong", result);

  result = doca_gpu_dev_rdma_commit_strong(rdma_gpu, connection_index);
  if (result != DOCA_SUCCESS)
    printf("Error doca_gpu_dev_rdma_commit_strong", result);

  __threadfence_block();
  __syncthreads();

  result = doca_gpu_dev_rdma_recv_wait_all(
      rdma_gpu_r, DOCA_GPU_RDMA_RECV_WAIT_FLAG_B, &num_ops, imm_val, conn_idx);
  if (result != DOCA_SUCCESS)
    printf("Error %d doca_gpu_dev_rdma_recv_wait_all\n", result);
}

extern "C" {

doca_error_t launch_expert_server(cudaStream_t stream,
                                  struct doca_gpu_dev_rdma *rdma_gpu,
                                  struct doca_gpu_buf_arr *buf_arr_hidden,
                                  struct doca_gpu_buf_arr *buf_arr_flag,
                                  uint32_t connection_index) {

  CHECK_CUDA(cudaGetLastError());
  kernel_server<<<1, 1, 0, stream>>>(rdma_gpu, buf_arr_hidden, buf_arr_flag,
                                     connection_index);
  CHECK_CUDA(cudaGetLastError());
  return DOCA_SUCCESS;
}
}