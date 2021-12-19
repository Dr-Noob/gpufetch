#include <cstddef>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <cerrno>
#include <cmath>

#include "../common/global.hpp"
#include "gpu.hpp"

#define STRING_YES        "Yes"
#define STRING_NO         "No"
#define STRING_NONE       "None"
#define STRING_MEGAHERZ   "MHz"
#define STRING_GIGAHERZ   "GHz"
#define STRING_KILOBYTES  "KB"
#define STRING_MEGABYTES  "MB"
#define STRING_GIGABYTES  "GB"

static const char *memtype_str[] = {
  /*[MEMTYPE_UNKNOWN] = */ STRING_UNKNOWN,
  /*[MEMTYPE_DDR3]    = */ "DDR3",
  /*[MEMTYPE_DDR4]    = */ "DDR4",
  /*[MEMTYPE_GDDR5]   = */ "GDDR5",
  /*[MEMTYPE_GDDR5X]  = */ "GDDR5X",
  /*[MEMTYPE_GDDR6]   = */ "GDDR6",
  /*[MEMTYPE_GDDR6X]  = */ "GDDR6X",
  /*[MEMTYPE_HBM2]    = */ "HBM2"
};

VENDOR get_gpu_vendor(struct gpu_info* gpu) {
  return gpu->vendor;
}

int32_t get_value_as_smallest_unit(char ** str, uint64_t value) {
  int32_t ret;
  int max_len = 10; // Max is 8 for digits, 2 for units
  *str = (char *) emalloc(sizeof(char)* (max_len + 1));

  if(value/1024 >= (1 << 20))
    ret = snprintf(*str, max_len, "%.4g " STRING_GIGABYTES, trunc((double)value/(1<<30)));
  else if(value/1024 >= (1 << 10))
    ret = snprintf(*str, max_len, "%.4g " STRING_MEGABYTES, trunc((double)value/(1<<20)));
  else
    ret = snprintf(*str, max_len, "%.4g " STRING_KILOBYTES, trunc((double)value/(1<<10)));

  return ret;
}

char* get_str_gpu_name(struct gpu_info* gpu) {
  return gpu->name;
}

char* get_freq_as_str_mhz(int64_t freq) {
  // Max 20 digits and 3 for 'MHz'
  uint32_t size = (20+1+3+1);
  assert(strlen(STRING_UNKNOWN) + 1 <= size);
  char* string = (char *) ecalloc(size, sizeof(char));

  if(freq == UNKNOWN_FREQ || freq < 0)
    snprintf(string, strlen(STRING_UNKNOWN)+1, STRING_UNKNOWN);
  else
    snprintf(string, size, "%ld " STRING_MEGAHERZ, freq);

  return string;
}

char* get_str_freq(struct gpu_info* gpu) {
  return get_freq_as_str_mhz(gpu->freq);
}

// TODO: Refactor
char* get_str_memory_size(struct gpu_info* gpu) {
  char* string;
  int32_t str_len = get_value_as_smallest_unit(&string, gpu->mem->size_bytes);

  if(str_len < 0) {
    printBug("get_value_as_smallest_unit: %s", strerror(errno));
    return NULL;
  }

  return string;
}

char* get_str_memory_type(struct gpu_info* gpu) {
  char* str = (char *) emalloc(sizeof(char) * (strlen(memtype_str[gpu->mem->type]) +1));
  strcpy(str, memtype_str[gpu->mem->type]);
  return str;
}

char* get_str_bus_width(struct gpu_info* gpu) {
  uint32_t size = 3+1+3+1;
  assert(strlen(STRING_UNKNOWN)+1 <= size);
  char* string = (char *) ecalloc(size, sizeof(char));

  sprintf(string, "%d bit", gpu->mem->bus_width);

  return string;
}

char* get_str_memory_clock(struct gpu_info* gpu) {
  return get_freq_as_str_mhz(gpu->mem->freq);
}

char* get_str_l2(struct gpu_info* gpu) {
  char* string;
  int32_t str_len = get_value_as_smallest_unit(&string, gpu->cach->L2->size);

  if(str_len < 0) {
    printBug("get_value_as_smallest_unit: %s", strerror(errno));
    return NULL;
  }

  return string;
}

char* get_str_peak_performance_generic(int64_t pp) {
  char* str;

  if(pp == -1) {
    str = (char *) emalloc(sizeof(char) * (strlen(STRING_UNKNOWN) + 1));
    strncpy(str, STRING_UNKNOWN, strlen(STRING_UNKNOWN) + 1);
    return str;
  }

  // 7 for digits (e.g, XXXX.XX), 7 for XFLOP/s
  double flopsd = (double) pp;
  uint32_t max_size = 7+1+7+1;
  str = (char *) ecalloc(max_size, sizeof(char));

  if(flopsd >= (double)1000000000000.0)
    snprintf(str, max_size, "%.2f TFLOP/s", flopsd/1000000000000);
  else if(flopsd >= 1000000000.0)
    snprintf(str, max_size, "%.2f GFLOP/s", flopsd/1000000000);
  else
    snprintf(str, max_size, "%.2f MFLOP/s", flopsd/1000000);

  return str;
}

char* get_str_peak_performance(struct gpu_info* gpu) {
  return get_str_peak_performance_generic(gpu->peak_performance);
}

char* get_str_peak_performance_tensor(struct gpu_info* gpu) {
  return get_str_peak_performance_generic(gpu->peak_performance_tcu);
}

char* get_str_generic(int32_t data) {
  // Largest int is 10, +1 for possible negative, +1 for EOL
  uint32_t max_size = 12;
  char* dummy = (char *) ecalloc(max_size, sizeof(char));
  snprintf(dummy, max_size, "%d", data);
  return dummy;
}
