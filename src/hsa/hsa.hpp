#ifndef __HSA_GPU__
#define __HSA_GPU__

#include "../common/gpu.hpp"

struct gpu_info* get_gpu_info_hsa(struct pci_dev *devices, int gpu_idx);

#endif
