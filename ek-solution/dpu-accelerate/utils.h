#pragma once

// --- Error Checking Wrappers ---
// Macro to wrap CUDA API calls and check for errors
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Macro to wrap cuBLAS API calls and check for errors
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error in %s at line %d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


#define CHECK_DOCA(call) do { \
    doca_error_t result = call; \
	if (result != DOCA_SUCCESS) { \
		DOCA_LOG_ERR("DOCA operation error", doca_error_get_descr(result)); \
        exit(EXIT_FAILURE); \
	} \
} while(0)






