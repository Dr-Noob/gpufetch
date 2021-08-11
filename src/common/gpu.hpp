#ifndef __GPU__
#define __GPU__

#include <stdint.h>
#include <stdbool.h>

enum {
  GPU_VENDOR_NVIDIA
};

typedef int32_t VENDOR;

struct gpu_info {
  VENDOR vendor;
  struct uarch* arch;
  char* name;
};


char* get_str_gpu_name(struct gpu_info* gpu);

#endif
