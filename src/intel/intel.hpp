#ifndef __INTEL_GPU__
#define __INTEL_GPU__

#include "../common/gpu.hpp"

struct gpu_info* get_gpu_info_intel();
bool print_gpu_intel(struct gpu_info* gpu);
char* get_str_eu(struct gpu_info* gpu);

#endif
