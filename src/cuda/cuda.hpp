#ifndef __CUDA_GPU__
#define __CUDA_GPU__

#include "../common/gpu.hpp"

struct gpu_info* get_gpu_info_cuda(int gpu_idx);
bool print_gpu_cuda(struct gpu_info* gpu);
char* get_str_sm(struct gpu_info* gpu);
char* get_str_cores_sm(struct gpu_info* gpu);
char* get_str_cuda_cores(struct gpu_info* gpu);
char* get_str_tensor_cores(struct gpu_info* gpu);

#endif
