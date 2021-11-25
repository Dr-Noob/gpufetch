#include <stdbool.h>
#include <cstddef>
#include <cstdlib>

#include "master.hpp"

#define MAX_GPUS 1000

struct gpu_list {
  struct gpu_info ** gpus;
  int num_gpus;
};

struct gpu_list* get_gpu_list() {
  bool valid = true;
  struct gpu_list* list = (struct gpu_list*) malloc(sizeof(struct gpu_list));
  list->num_gpus = 0;
  list->gpus = (struct gpu_info**) malloc(sizeof(struct info*) * MAX_GPUS);

#ifdef ENABLE_CUDA_BACKEND
  int idx = 0;

  while(valid) {
    list->gpus[idx] = get_gpu_info_cuda(idx);
    if(list->gpus[idx] != NULL) idx++;
    else valid = false;
  }

  list->num_gpus += idx;
#endif

#ifdef ENABLE_INTEL_BACKEND
  list->gpus[idx] = get_gpu_info_intel();
  if(list->gpus[idx] != NULL) list->num_gpus++;
#endif

  return list;
}

bool print_gpus_list(struct gpu_list* list) {
  return false;
}

struct gpu_info* get_gpu_info(struct gpu_list* list, int idx) {
  return list->gpus[idx];
}
