#ifndef __GPU__
#define __GPU__

#include <stdint.h>
#include <stdbool.h>

enum {
  GPU_VENDOR_NVIDIA
};

enum {
  MEMTYPE_GDDR6
};

typedef int32_t VENDOR;
typedef int32_t MEMTYPE;

struct cach {
  int32_t size;
  uint8_t num_caches;
  bool exists;
};

struct cache {
  struct cach* L1;
  struct cach* L2;
};

struct topology {
  int32_t shared_mp;
  int32_t cores_per_mp;
  int32_t cuda_cores;
};

struct memory {
  int32_t size_bytes;
  MEMTYPE type;
};

struct gpu_info {
  VENDOR vendor;
  struct uarch* arch;
  char* name;
  int64_t freq;
  struct topology* topo;
  struct memory* mem;
  struct cache* cach;
  int64_t peak_performance;
};

char* get_str_gpu_name(struct gpu_info* gpu);
char* get_str_freq(struct gpu_info* gpu);
char* get_str_memory_size(struct gpu_info* gpu);
char* get_str_memory_type(struct gpu_info* gpu);
char* get_str_l1(struct gpu_info* gpu);
char* get_str_l2(struct gpu_info* gpu);
char* get_str_peak_performance(struct gpu_info* gpu);

#endif
