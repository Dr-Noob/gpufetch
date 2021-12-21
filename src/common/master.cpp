#include <cstdlib>
#include <cstdio>

#include "global.hpp"
#include "colors.hpp"
#include "master.hpp"
#include "../cuda/cuda.hpp"
#include "../intel/intel.hpp"

#define MAX_GPUS 1000

struct gpu_list {
  struct gpu_info ** gpus;
  int num_gpus;
};

struct gpu_list* get_gpu_list() {
  int idx = 0;
  struct gpu_list* list = (struct gpu_list*) malloc(sizeof(struct gpu_list));
  list->num_gpus = 0;
  list->gpus = (struct gpu_info**) malloc(sizeof(struct info*) * MAX_GPUS);

#ifdef BACKEND_CUDA
  bool valid = true;

  while(valid) {
    list->gpus[idx] = get_gpu_info_cuda(idx);
    if(list->gpus[idx] != NULL) idx++;
    else valid = false;
  }

  list->num_gpus += idx;
#endif

#ifdef BACKEND_INTEL
  list->gpus[idx] = get_gpu_info_intel();
  if(list->gpus[idx] != NULL) list->num_gpus++;
#endif

  return list;
}

bool print_gpus_list(struct gpu_list* list) {
  for(int i=0; i < list->num_gpus; i++) {
    printf("GPU %d: ", i);
    if(list->gpus[i]->vendor == GPU_VENDOR_NVIDIA) {
      #ifdef BACKEND_CUDA
        print_gpu_cuda(list->gpus[i]);
      #endif
    }
    else if(list->gpus[i]->vendor == GPU_VENDOR_INTEL) {
      #ifdef BACKEND_INTEL
        print_gpu_intel(list->gpus[i]);
      #endif
    }
  }

  return true;
}

void print_enabled_backends() {
  printf("- CUDA backend:  ");
#ifdef BACKEND_CUDA
  printf("%sON%s\n", C_FG_GREEN, C_RESET);
#else
  printf("%sOFF%s\n", C_FG_RED, C_RESET);
#endif

  printf("- Intel backend: ");
#ifdef BACKEND_INTEL
  printf("%sON%s\n", C_FG_GREEN, C_RESET);
#else
  printf("%sOFF%s\n", C_FG_RED, C_RESET);
#endif
}

int get_num_gpus_available(struct gpu_list* list) {
  return list->num_gpus;
}

struct gpu_info* get_gpu_info(struct gpu_list* list, int idx) {
  if(idx >= list->num_gpus || idx < 0) {
    printErr("Specified GPU index is out of range: %d", idx);
    return NULL;
  }
  return list->gpus[idx];
}

