#ifndef __CUDA__
#define __CUDA__

#include "../common/gpu.hpp"

struct gpu_info* get_gpu_info(int gpu_idx);
char* get_str_sm(struct gpu_info* gpu);
char* get_str_cores_sm(struct gpu_info* gpu);
char* get_str_cuda_cores(struct gpu_info* gpu);

#endif
