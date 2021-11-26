#include <stdio.h>

#include "intel.hpp"
#include "uarch.hpp"
#include "../common/pci.hpp"
#include "../common/global.hpp"

struct gpu_info* get_gpu_info_intel() {
  struct gpu_info* gpu = (struct gpu_info*) emalloc(sizeof(struct gpu_info));
  gpu->vendor = GPU_VENDOR_INTEL;

  return gpu;
}

bool print_gpu_intel(struct gpu_info* gpu) {
  if(gpu->vendor != GPU_VENDOR_INTEL) return false;

  printf("Intel ???\n");

  return true;
}
