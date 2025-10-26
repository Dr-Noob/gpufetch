#ifndef __HSA_GPU__
#define __HSA_GPU__

#include "../common/gpu.hpp"

struct gpu_info* get_gpu_info_hsa(int gpu_idx);
char* get_str_cu(struct gpu_info* gpu);
char* get_str_xcds(struct gpu_info* gpu);
char* get_str_matrix_cores(struct gpu_info* gpu);

#endif
