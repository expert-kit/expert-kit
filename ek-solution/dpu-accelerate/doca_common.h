

struct rdma_resources {
  struct rdma_config *cfg;
  struct doca_dev *doca_device;
  struct doca_gpu *gpudev;
  struct doca_rdma *rdma;
  struct doca_gpu_dev_rdma *gpu_rdma;
  struct doca_ctx *rdma_ctx;
  struct doca_pe *pe;
  const void *connection_details;
  size_t conn_det_len;

  struct doca_rdma_addr *cm_addr;
  struct doca_rdma_connection *connection;
  bool connection_established;
  bool connection_error;
  bool server_listen_active;

  struct doca_rdma_connection *connection2;
  bool connection2_established;
  bool connection2_error;
};

struct rdma_config {
  char device_name[1024];
  char gpu_pcie_addr[1024];
  char server_ip_addr[1024];
  uint32_t cm_port;
  char cm_addr[1024];
};
