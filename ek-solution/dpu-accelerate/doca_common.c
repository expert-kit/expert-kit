#include "utils.h"
#include "assert.h"

doca_error_t create_rdma_resources(struct rdma_config *cfg,
				   const uint32_t rdma_permissions,
				   struct rdma_resources *resources)
{
	union doca_data ctx_user_data = {0};
	doca_error_t result, tmp_result;
	resources->cfg = cfg;

    CHECK_DOCA(open_doca_device_with_ibdev_name((const uint8_t *)(cfg->device_name),
						  strlen(cfg->device_name),
						  wrapper_doca_rdma_cap_task_write_is_supported,
						  &(resources->doca_device)));
	CHECK_DOCA(doca_gpu_create(cfg->gpu_pcie_addr, &(resources->gpudev)));
	CHECK_DOCA(doca_rdma_create(resources->doca_device, &(resources->rdma)));
	resources->rdma_ctx = doca_rdma_as_ctx(resources->rdma);
    assert(resources->rdma_ctx != NULL);
	CHECK_DOCA(doca_rdma_set_permissions(resources->rdma, rdma_permissions));
	if (cfg->is_gid_index_set) {
		CHECK_DOCA(doca_rdma_set_gid_index(resources->rdma, cfg->gid_index));
	}

	CHECK_DOCA(doca_rdma_set_send_queue_size(resources->rdma, RDMA_SEND_QUEUE_SIZE));
	CHECK_DOCA(doca_ctx_set_datapath_on_gpu(resources->rdma_ctx, resources->gpudev));
	CHECK_DOCA(doca_rdma_set_recv_queue_size(resources->rdma, RDMA_RECV_QUEUE_SIZE));
	CHECK_DOCA(doca_rdma_set_grh_enabled(resources->rdma, true));
    CHECK_DOCA(doca_pe_create(&(resources->pe)));
    CHECK_DOCA(doca_pe_connect_ctx(resources->pe, resources->rdma_ctx));
    CHECK_DOCA(doca_rdma_set_max_num_connections(resources->rdma, 2));
    CHECK_DOCA(doca_rdma_set_connection_state_callbacks(resources->rdma,
                                rdma_cm_connect_request_cb,
                                rdma_cm_connect_established_cb,
                                rdma_cm_connect_failure_cb,
                                rdma_cm_disconnect_cb));
	ctx_user_data.ptr = resources;
    CHECK_DOCA(doca_ctx_set_user_data(resources->rdma_ctx, ctx_user_data));
	CHECK_DOCA(doca_ctx_start(resources->rdma_ctx));
	return DOCA_SUCCESS;
}



void init_doca(struct rdma_config *cfg)
{
	cfg->cm_port = 8088;
	cfg->cm_addr_type = DOCA_RDMA_ADDR_TYPE_IPv4;
	cfg->cm_addr = "192.168.102.1";
	cfg->device_name= "mlx5_1";
	cfg->gpu_pcie_addr = "e1:00.0";
}
