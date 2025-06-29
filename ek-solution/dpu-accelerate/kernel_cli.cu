
__global__ void forward_hidden(struct doca_gpu_dev_rdma *rdma_gpu,
                               struct doca_gpu_buf_arr *buf_hidden,
                               struct doca_gpu_buf_arr *buf_result,
                               struct doca_gpu_buf_arr *buf_flag,
                               uint32_t *exit_flag) {
  doca_error_t result;
  struct doca_gpu_buf *lbuf_hidden;
  struct doca_gpu_buf *lbuf_result;
  struct doca_gpu_buf *lbuf_flag;
  int buf_index = 0;
  uint32_t num_ops;
  uint32_t imm_val[2];
  uint32_t conn_idx[2];
  uintptr_t laddr_f;
  struct doca_gpu_dev_rdma_r *rdma_gpu_r;

  doca_gpu_dev_buf_get_buf(buf_hidden, buf_index, &lbuf_hidden);
  doca_gpu_dev_buf_get_buf(buf_result, 0, &lbuf_result);
  doca_gpu_dev_buf_get_buf(buf_flag, 0, &lbuf_flag);
  doca_gpu_dev_buf_get_addr(lbuf_flag, &laddr_f);

  printf("wait for recv posted");
  while (DOCA_GPUNETIO_VOLATILE(((uint8_t *)laddr_f)) != 1)
    __threadfence_block();

  if (threadIdx.x == 0) {
    // send hidden vector to server
    doca_gpu_dev_rdma_send_strong(rdma_gpu, 0, lbuf_hidden, 0, LEN_HIDDEN, 0,
                                  DOCA_GPU_RDMA_SEND_FLAG_IMM);

    result = doca_gpu_dev_rdma_commit_strong(rdma_gpu, connection_index);

    if (result != DOCA_SUCCESS)
      printf("Error %d doca_gpu_dev_rdma_push\n", result);

    result = doca_gpu_dev_rdma_wait_all(rdma_gpu, &num_ops);
    if (result != DOCA_SUCCESS)
      printf("Error %d doca_gpu_dev_rdma_wait_all\n", result);

  } else {
    // receive hidden vector from server
    result = doca_gpu_dev_rdma_get_recv(rdma_gpu, &rdma_gpu_r);
    if (result != DOCA_SUCCESS)
      printf("Error %d doca_gpu_dev_buf_get_buf\n", result);

    result = doca_gpu_dev_rdma_recv_strong(rdma_gpu_r, lbuf_result, HIDDEN_SIZE,
                                           (threadIdx.x * HIDDEN_SIZE), 0);

    if (result != DOCA_SUCCESS)
      printf("Error %d doca_gpu_dev_rdma_recv_wait_all\n", result);
    result = doca_gpu_dev_rdma_recv_wait_all(rdma_gpu_r,
                                             DOCA_GPU_RDMA_RECV_WAIT_FLAG_B,
                                             &num_ops, imm_val, conn_idx);
    if (result != DOCA_SUCCESS)
      printf("Error %d doca_gpu_dev_rdma_push\n", result);
  }

  while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0)
    ;
  __syncthreads();
}

extern "C" {
doca_error_t launch_forward_hidden(cudaStream_t stream,
                                   struct doca_gpu_dev_rdma *rdma_gpu,
                                   struct doca_gpu_buf_arr *hidden_input,
                                   struct doca_gpu_buf_arr *hidden_remote,
                                   struct doca_gpu_buf_arr *buf_flag,
                                   uint32_t *exit_flag) {

  CHECK_CUDA(cudaGetLastError());
  kernel_client<<<1, 2, 0, stream>>>(rdma_gpu, hidden_input, hidden_remote,
                                     exit_flag);
  CHECK_CUDA(cudaGetLastError());

  return DOCA_SUCCESS;
}
}