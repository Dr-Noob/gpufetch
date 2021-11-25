#ifndef __GPU_LIST__
#define __GPU_LIST__

#include "gpu.hpp"

struct gpu_list;

struct gpu_list* get_gpu_list();
bool print_gpus_list(struct gpu_list* list);
struct gpu_info* get_gpu_info(struct gpu_list* list, int idx);

#endif
