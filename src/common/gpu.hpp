#ifndef __GPU__
#define __GPU__

#include <cstdint>

#define UNKNOWN_FREQ -1

enum {
  GPU_VENDOR_NVIDIA,
  GPU_VENDOR_AMD,
  GPU_VENDOR_INTEL
};

enum {
  MEMTYPE_UNKNOWN,
  MEMTYPE_DDR3,
  MEMTYPE_DDR4,
  MEMTYPE_GDDR5,
  MEMTYPE_GDDR5X,
  MEMTYPE_GDDR6,
  MEMTYPE_GDDR6X,
  MEMTYPE_HBM2
};

typedef int32_t VENDOR;
typedef int32_t MEMTYPE;

struct cach {
  int32_t size;
  uint8_t num_caches;
  bool exists;
};

struct cache {
  struct cach* L2;
};

// CUDA topology
struct topology_c {
  int32_t streaming_mp;
  int32_t cores_per_mp;
  int32_t cuda_cores;
  int32_t tensor_cores;
};

// HSA topology
struct topology_h {
  int32_t compute_units;
  int32_t num_shader_engines;
  int32_t simds_per_cu;
  int32_t num_xcc;
};

// Intel topology
struct topology_i {
  int32_t slices;
  int32_t subslices;
  int32_t eu_subslice;
};

struct memory {
  int64_t size_bytes;
  MEMTYPE type;
  int32_t bus_width;
  int32_t freq;
  int32_t clk_mul; // clock multiplier
  int32_t lds_size; // HSA specific for now
};

struct gpu_info {
  int32_t idx;
  VENDOR vendor;
  struct uarch* arch;
  char* name;
  int64_t freq;
  struct pci* pci;
  int64_t peak_performance;
  // CUDA specific
  int64_t peak_performance_tcu;
  struct memory* mem;
  struct cache* cach;
  struct topology_c* topo_c;
  // HSA specific
  struct topology_h* topo_h;
  // Intel specific
  struct topology_i* topo_i;
};

VENDOR get_gpu_vendor(struct gpu_info* gpu);
char* get_str_gpu_name(struct gpu_info* gpu);
char* get_str_freq(struct gpu_info* gpu);
char* get_str_memory_size(struct gpu_info* gpu);
char* get_str_memory_type(struct gpu_info* gpu);
char* get_str_bus_width(struct gpu_info* gpu);
char* get_str_lds_size(struct gpu_info* gpu);
char* get_str_memory_clock(struct gpu_info* gpu);
char* get_str_l2(struct gpu_info* gpu);
char* get_str_peak_performance(struct gpu_info* gpu);
char* get_str_peak_performance_tensor(struct gpu_info* gpu);
char* get_str_generic(int32_t data);

#endif
