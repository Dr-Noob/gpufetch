#include <stdio.h>
#include <string.h>

#include "intel.hpp"
#include "uarch.hpp"
#include "../common/pci.hpp"
#include "../common/global.hpp"

struct gpu_info* get_gpu_info_intel() {
  struct gpu_info* gpu = (struct gpu_info*) emalloc(sizeof(struct gpu_info));
  const char* name = "UHD Graphics XXX";

  gpu->vendor = GPU_VENDOR_INTEL;
  gpu->name = (char *) emalloc(sizeof(char) * (strlen(name) + 1));
  strcpy(gpu->name, name);

  return gpu;
}

bool print_gpu_intel(struct gpu_info* gpu) {
  if(gpu->vendor != GPU_VENDOR_INTEL) return false;

  printf("Intel %s\n", gpu->name);

  return true;
}
