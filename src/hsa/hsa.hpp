#ifndef __HSA_GPU__
#define __HSA_GPU__

#include "../common/gpu.hpp"
// #define CUDA_DRIVER_START_WARNING "Waiting for CUDA driver to start..."

struct gpu_info* get_gpu_info_hsa(struct pci_dev *devices, int gpu_idx);

#endif
