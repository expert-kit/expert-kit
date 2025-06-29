
static struct rdma_ctx resources = {0};

void init_rdma_cfg(struct rdma_cfg *arg) {
  cfg.is_server = true;
  cfg.cm_port = DEFAULT_CM_PORT;
  cfg.cm_addr_type = DOCA_RDMA_ADDR_TYPE_IPv4;
  cfg.gid_index = 0;
  cfg.use_rdma_cm = false;
}

int init_doca_srv() {
  struct rdma_config cfg = {0};
  init_rdma_cfg(&cfg);
}

struct rdma_mmap_obj mm_srv_hidden = {0};
uint8_t *buf_hidden_gpu;
uint8_t *buf_hidden_cpu;
uint8_t *buf_result;

void oob_exchange(int oob_sock_fd) {
  size_t result_len;
  if (send(oob_sock_fd, &mm_srv_hidden.export_len, sizeof(size_t), 0) < 0) {
    goto error;
  }
  if (send(oob_sock_fd, mm_srv_hidden.rdma_export, mm_srv_hidden.export_len,
           0) < 0) {
    goto error;
  }

  DOCA_LOG_INFO("Receive client mmap F export");
  if (recv(oob_sock_fd, &result_len, sizeof(size_t), 0) < 0) {
    goto error;
  }

  buf_result = calloc(1, result_len);
  if (server_remote_export_F == NULL) {
    DOCA_LOG_ERR("Failed to allocate memory for remote mmap export");
    goto error;
  }

  if (recv(oob_sock_fd, buf_result, result_len, 0) < 0) {
    DOCA_LOG_ERR("Failed to receive remote connection details");
    goto error;
  }

error:
  printf("Error in oob_exchange\n");
  exit(1)
}

static doca_error_t create_mem(int oob_sock_fd,
                               struct rdma_resources *resources, int conn_idx,
                               cudaStream_t stream) {
  void *server_remote_export_F = NULL;
  size_t server_remote_export_F_len;
  doca_error_t result;
  cudaError_t cuda_err;

  DOCA_CHECK(doca_gpu_mem_alloc(
      resources->gpudev, (size_t)GPU_BUF_NUM * GPU_BUF_SIZE_A, 4096,
      DOCA_GPU_MEM_TYPE_GPU_CPU, (void **)&server_local_buf_A_gpu[conn_idx],
      (void **)&server_local_buf_A_cpu[conn_idx]));
  CHECK_CUDA(cudaMemsetAsync(server_local_buf_A_gpu[conn_idx], 0x1,
                             GPU_BUF_NUM * GPU_BUF_SIZE_A, stream));

  mm_srv_hidden.doca_device = resources->doca_device;
  mm_srv_hidden.permissions = access_params;
  mm_srv_hidden.memrange_addr = server_local_buf_A_gpu[conn_idx];
  mm_srv_hidden.memrange_len = (size_t)GPU_BUF_NUM * GPU_BUF_SIZE_A;
  DOCA_CHECK(create_mmap(&mm_srv_hidden));

  oob_exchange(oob_sock_fd);

  DOCA_CHECK(doca_mmap_create_from_export(
      NULL, server_remote_export_F, server_remote_export_F_len,
      resources->doca_device, &server_remote_mmap_F[conn_idx]));

  server_local_buf_arr_A[conn_idx].gpudev = resources->gpudev;
  server_local_buf_arr_A[conn_idx].mmap =
      server_local_mmap_obj_A[conn_idx].mmap;
  server_local_buf_arr_A[conn_idx].num_elem = GPU_BUF_NUM;
  server_local_buf_arr_A[conn_idx].elem_size = GPU_BUF_SIZE_A;

  DOCA_CHECK(create_buf_arr_on_gpu(&server_local_buf_arr_A[conn_idx]));

  server_remote_buf_arr_F[conn_idx].gpudev = resources->gpudev;
  server_remote_buf_arr_F[conn_idx].mmap = server_remote_mmap_F[conn_idx];
  server_remote_buf_arr_F[conn_idx].num_elem = 1;
  server_remote_buf_arr_F[conn_idx].elem_size =
      (size_t)(GPU_BUF_NUM * GPU_BUF_SIZE_F);

  DOCA_CHECK(create_buf_arr_on_gpu(&server_remote_buf_arr_F[conn_idx]));

  free(server_remote_export_F);

  return DOCA_SUCCESS;

error:
  if (server_remote_export_F)
    free(server_remote_export_F);

  return result;
}

doca_error_t srv_main(struct rdma_cfg *cfg) {
  struct doca_rdma_connection *connection = NULL;
  const uint32_t rdma_permissions = access_params;
  doca_error_t result, tmp_result;
  void *remote_conn_details = NULL;
  size_t remote_conn_details_len = 0;
  cudaError_t cuda_ret;
  int ret = 0;
  struct timespec ts = {
      .tv_sec = 0,
      .tv_nsec = SLEEP_IN_NANOS,
  };

  CHECK_DOCA(create_rdma_resources(cfg, rdma_permissions, &resources));
  CHECK_DOCA(doca_rdma_get_gpu_handle(resources.rdma, &(resources.gpu_rdma)));
  ret = oob_connection_server_setup(&oob_sock_fd, &oob_client_sock);
  assert(ret >= 0);
  CHECK_DOCA(doca_rdma_start_listen_to_port(resources.rdma, cfg->cm_port));

  resources.server_listen_active = true;
  while ((!resources.connection_established) && (!resources.connection_error)) {
    if (doca_pe_progress(resources.pe) == 0)
      nanosleep(&ts, &ts);
  }
  if (resources.connection_error) {
    result = DOCA_ERROR_CONNECTION_ABORTED;
    goto close_connection;
  }

  CHECK_CUDA(cudaStreamCreateWithFlags(&cstream, cudaStreamNonBlocking));
  CHECK_DOCA(create_memory_local_remote_server(oob_client_sock, &resources, 0,
                                               cstream));
  CHECK_DOCA(launch_kernel_server(cstream, resources.gpu_rdma,
                                  server_local_buf_arr_A[0].gpu_buf_arr,
                                  server_remote_buf_arr_F[0].gpu_buf_arr, 0));
  cudaStreamSynchronize(cstream);
  oob_connection_server_close(oob_sock_fd, oob_client_sock);
  destroy_memory_local_remote_server(&resources);
  CHECK_DOCA(destroy_rdma_resources(&resources));
  return DOCA_SUCCESS;
}