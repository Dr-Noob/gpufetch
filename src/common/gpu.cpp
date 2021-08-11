#include "gpu.hpp"
#include <cstddef>

char* get_str_gpu_name(struct gpu_info* gpu) {
  return gpu->name;
}

char* get_str_freq(struct gpu_info* gpu) {
  return NULL;
}

char* get_str_memory_size(struct gpu_info* gpu) {
  return NULL;
}

char* get_str_memory_type(struct gpu_info* gpu) {
  return NULL;
}

char* get_str_l1(struct gpu_info* gpu) {
  return NULL;
}

char* get_str_l2(struct gpu_info* gpu) {
  return NULL;
}

char* get_str_peak_performance(struct gpu_info* gpu) {
  return NULL;
}
