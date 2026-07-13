
struct rdma_mmap_obj mm_cli_hidden = {0};
uint8_t *buf_hidden_gpu;
uint8_t *buf_hidden_cpu;
uint8_t *buf_result_gpu;

static doca_error_t create_mm_cli(int oob_sock_fd,
                                  struct rdma_resources *resources,
                                  cudaStream_t stream) {

  void *client_remote_export_A = NULL;
  size_t srv_hidden_len;
  doca_error_t result;
  cudaError_t cuda_err;

  CHECK_DOCA(doca_gpu_mem_alloc(
      resources->gpudev, (size_t)GPU_BUF_NUM * GPU_BUF_SIZE_B, 4096,
      DOCA_GPU_MEM_TYPE_GPU_CPU, (void **)&buf_hidden_gpu,
      (void **)&buf_hidden_cpu));

  CHECK_CUDA(cudaMemsetAsync(buf_hidden_gpu, 0x2, GPU_BUF_NUM * GPU_BUF_SIZE_B,
                             stream));

  mm_cli_hidden.doca_device = resources->doca_device;
  mm_cli_hidden.permissions = access_params;
  mm_cli_hidden.memrange_addr = buf_hidden_gpu;
  mm_cli_hidden.memrange_len = (size_t)GPU_BUF_NUM * GPU_BUF_SIZE_B;

  DOCA_LOG_INFO("Create local client mmap B context");
  CHECK_DOCA(create_mmap(&mm_cli_hidden));

  DOCA_LOG_INFO("Receive remote mmap A export from server");
  if (recv(oob_sock_fd, &srv_hidden_len, sizeof(size_t), 0) < 0) {
    DOCA_LOG_ERR("Failed to receive remote connection details");
    result = DOCA_ERROR_CONNECTION_ABORTED;
    goto error;
  }

  client_remote_export_A = calloc(1, client_remote_export_A_len);
  if (client_remote_export_A == NULL) {
    DOCA_LOG_ERR("Failed to allocate memory for remote mmap export");
    result = DOCA_ERROR_NO_MEMORY;
    goto error;
  }

  if (recv(oob_sock_fd, client_remote_export_A, client_remote_export_A_len, 0) <
      0) {
    DOCA_LOG_ERR("Failed to receive remote connection details");
    result = DOCA_ERROR_CONNECTION_ABORTED;
    goto error;
  }

  result = doca_mmap_create_from_export(
      NULL, client_remote_export_A, client_remote_export_A_len,
      resources->doca_device, &client_remote_mmap_A[conn_idx]);

  /* Send client local F */
  DOCA_LOG_INFO("Send exported mmap F to remote server");
  if (send(oob_sock_fd, &client_local_mmap_obj_F[conn_idx].export_len,
           sizeof(size_t), 0) < 0) {
    DOCA_LOG_ERR("Failed to send exported mmap");
    result = DOCA_ERROR_CONNECTION_ABORTED;
    goto error;
  }

  if (send(oob_sock_fd, client_local_mmap_obj_F[conn_idx].rdma_export,
           client_local_mmap_obj_F[conn_idx].export_len, 0) < 0) {
    DOCA_LOG_ERR("Failed to send exported mmap");
    result = DOCA_ERROR_CONNECTION_ABORTED;
    goto error;
  }

  /* create local and remote buf arrays */
  client_local_buf_arr_B[conn_idx].gpudev = resources->gpudev;
  client_local_buf_arr_B[conn_idx].mmap =
      client_local_mmap_obj_B[conn_idx].mmap;
  client_local_buf_arr_B[conn_idx].num_elem = GPU_BUF_NUM;
  client_local_buf_arr_B[conn_idx].elem_size = GPU_BUF_SIZE_B;

  /* create local buf array object */
  DOCA_LOG_INFO("Create local DOCA buf array context B");
  result = create_buf_arr_on_gpu(&client_local_buf_arr_B[conn_idx]);
  if (result != DOCA_SUCCESS) {
    DOCA_LOG_ERR("Function create_buf_arr_on_gpu failed: %s",
                 doca_error_get_descr(result));
    goto error;
  }

  client_local_buf_arr_C[conn_idx].gpudev = resources->gpudev;
  client_local_buf_arr_C[conn_idx].mmap =
      client_local_mmap_obj_C[conn_idx].mmap;
  client_local_buf_arr_C[conn_idx].num_elem = GPU_BUF_NUM;
  client_local_buf_arr_C[conn_idx].elem_size = GPU_BUF_SIZE_C;

  /* create local buf array object */
  DOCA_LOG_INFO("Create local DOCA buf array context C");
  result = create_buf_arr_on_gpu(&client_local_buf_arr_C[conn_idx]);
  if (result != DOCA_SUCCESS) {
    doca_buf_arr_destroy(client_local_buf_arr_B[conn_idx].buf_arr);
    DOCA_LOG_ERR("Function create_buf_arr_on_gpu failed: %s",
                 doca_error_get_descr(result));
    goto error;
  }

  client_local_buf_arr_F[conn_idx].gpudev = resources->gpudev;
  client_local_buf_arr_F[conn_idx].mmap =
      client_local_mmap_obj_F[conn_idx].mmap;
  client_local_buf_arr_F[conn_idx].num_elem = 1;
  client_local_buf_arr_F[conn_idx].elem_size =
      (size_t)(GPU_BUF_NUM * GPU_BUF_SIZE_F);

  /* create local buf array object */
  DOCA_LOG_INFO("Create local DOCA buf array context F");
  result = create_buf_arr_on_gpu(&client_local_buf_arr_F[conn_idx]);
  if (result != DOCA_SUCCESS) {
    doca_buf_arr_destroy(client_local_buf_arr_B[conn_idx].buf_arr);
    DOCA_LOG_ERR("Function create_buf_arr_on_gpu failed: %s",
                 doca_error_get_descr(result));
    goto error;
  }

  client_remote_buf_arr_A[conn_idx].gpudev = resources->gpudev;
  client_remote_buf_arr_A[conn_idx].mmap = client_remote_mmap_A[conn_idx];
  client_remote_buf_arr_A[conn_idx].num_elem = GPU_BUF_NUM;
  client_remote_buf_arr_A[conn_idx].elem_size = GPU_BUF_SIZE_A;

  /* create remote buf array object */
  DOCA_LOG_INFO("Create remote DOCA buf array context");
  result = create_buf_arr_on_gpu(&client_remote_buf_arr_A[conn_idx]);
  if (result != DOCA_SUCCESS) {
    DOCA_LOG_ERR("Function create_buf_arr_on_gpu failed: %s",
                 doca_error_get_descr(result));
    doca_buf_arr_destroy(client_local_buf_arr_B[conn_idx].buf_arr);
    doca_buf_arr_destroy(client_local_buf_arr_C[conn_idx].buf_arr);
    goto error;
  }

  free(client_remote_export_A);

  return DOCA_SUCCESS;

error:
  if (client_remote_export_A)
    free(client_remote_export_A);

  return result;
}

doca_error_t rdma_write_client(struct rdma_config *cfg) {
  struct doca_rdma_connection *connection = NULL;
  const uint32_t rdma_permissions = access_params;
  doca_error_t result, temp_result;
  cudaError_t cuda_ret;
  void *remote_conn_details = NULL;
  size_t remote_conn_details_len = 0;
  int ret = 0;
  union doca_data connection_data;
  uint32_t *cpu_exit_flag;
  uint32_t *gpu_exit_flag;
  struct timespec ts = {
      .tv_sec = 0,
      .tv_nsec = SLEEP_IN_NANOS,
  };

  /* Allocate resources */
  result = create_rdma_resources(cfg, rdma_permissions, &resources);
  if (result != DOCA_SUCCESS) {
    DOCA_LOG_ERR("Failed to allocate RDMA resources: %s",
                 doca_error_get_descr(result));
    return result;
  }

  /* Get GPU RDMA handle */
  result = doca_rdma_get_gpu_handle(resources.rdma, &(resources.gpu_rdma));
  if (result != DOCA_SUCCESS) {
    DOCA_LOG_ERR("Failed to get RDMA GPU handler: %s",
                 doca_error_get_descr(result));
    goto destroy_resources;
  }

  /* Setup OOB connection */
  ret = oob_connection_client_setup(cfg->server_ip_addr, &oob_sock_fd);
  if (ret < 0) {
    DOCA_LOG_ERR("Failed to setup OOB connection with remote peer");
    result = DOCA_ERROR_CONNECTION_ABORTED;
    goto destroy_resources;
  }

  result = doca_rdma_addr_create(cfg->cm_addr_type, cfg->cm_addr, cfg->cm_port,
                                 &resources.cm_addr);
  if (result != DOCA_SUCCESS) {
    DOCA_LOG_ERR("Failed to create rdma cm connection address %s",
                 doca_error_get_descr(result));
    goto close_connection;
  }

  connection_data.ptr = (void *)&resources;
  result = doca_rdma_connect_to_addr(resources.rdma, resources.cm_addr,
                                     connection_data);
  if (result != DOCA_SUCCESS) {
    DOCA_LOG_ERR("Client failed to call doca_rdma_connect_to_addr %s",
                 doca_error_get_descr(result));
    goto close_connection;
  }

  DOCA_LOG_INFO("Client is waiting for a connection establishment");
  /* Wait for a new connection */
  while ((!resources.connection_established) && (!resources.connection_error)) {
    if (doca_pe_progress(resources.pe) == 0)
      nanosleep(&ts, &ts);
  }

  if (resources.connection_error) {
    DOCA_LOG_ERR("Failed to connect to remote peer, connection error");
    result = DOCA_ERROR_CONNECTION_ABORTED;
    goto close_connection;
  }

  DOCA_LOG_INFO("Client - Connection 1 is established");

  result = create_memory_local_remote_client(oob_sock_fd, &resources, 0, 0);
  if (result != DOCA_SUCCESS) {
    DOCA_LOG_ERR("Function create_memory_local_remote_client failed: %s",
                 doca_error_get_descr(result));
    goto close_connection;
  }

  result = doca_gpu_mem_alloc(resources.gpudev, sizeof(uint32_t), 4096,
                              DOCA_GPU_MEM_TYPE_GPU_CPU,
                              (void **)&gpu_exit_flag, (void **)&cpu_exit_flag);
  if (result != DOCA_SUCCESS || gpu_exit_flag == NULL ||
      cpu_exit_flag == NULL) {
    DOCA_LOG_ERR("Function doca_gpu_mem_alloc returned %s",
                 doca_error_get_descr(result));
    goto close_connection;
  }
  cpu_exit_flag[0] = 0;

  cuda_ret = cudaStreamCreateWithFlags(&cstream, cudaStreamNonBlocking);
  if (cuda_ret != cudaSuccess) {
    DOCA_LOG_ERR("Function cudaStreamCreateWithFlags error %d", cuda_ret);
    result = DOCA_ERROR_DRIVER;
    goto close_connection;
  }

  /* First client kernel on default CUDA stream */
  result = kernel_write_client(
      0, resources.gpu_rdma, client_local_buf_arr_B[0].gpu_buf_arr,
      client_local_buf_arr_C[0].gpu_buf_arr,
      client_local_buf_arr_F[0].gpu_buf_arr,
      client_remote_buf_arr_A[0].gpu_buf_arr, 0, gpu_exit_flag);
  if (result != DOCA_SUCCESS) {
    DOCA_LOG_ERR("Function kernel_write_client failed: %s",
                 doca_error_get_descr(result));
    goto close_connection;
  }

  DOCA_GPUNETIO_VOLATILE(*cpu_exit_flag) = 1;
  cudaStreamSynchronize(0);

  cudaStreamSynchronize(cstream);
  cudaStreamDestroy(cstream);
  oob_connection_client_close(oob_sock_fd);
  destroy_memory_local_remote_client(&resources);
  CHECK_DOCA(destroy_rdma_resources(&resources));

  return DOCA_SUCCESS;
}