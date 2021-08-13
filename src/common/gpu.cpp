#include <cstddef>
#include <cstring>
#include <cstdio>
#include <cassert>

#include "../common/global.hpp"
#include "gpu.hpp"

#define STRING_YES        "Yes"
#define STRING_NO         "No"
#define STRING_NONE       "None"
#define STRING_MEGAHERZ   "MHz"
#define STRING_GIGAHERZ   "GHz"
#define STRING_KILOBYTES  "KB"
#define STRING_MEGABYTES  "MB"

char* get_str_gpu_name(struct gpu_info* gpu) {
  return gpu->name;
}

char* get_str_freq(struct gpu_info* gpu) {
  // Max 5 digits and 3 for '(M/G)Hz'
  uint32_t size = (5+1+3+1);
  assert(strlen(STRING_UNKNOWN)+1 <= size);
  char* string = (char *) ecalloc(size, sizeof(char));

  if(gpu->freq == UNKNOWN_FREQ || gpu->freq < 0)
    snprintf(string,strlen(STRING_UNKNOWN)+1, STRING_UNKNOWN);
  else if(gpu->freq >= 1000)
    snprintf(string,size,"%.3f "STRING_GIGAHERZ, (float)(gpu->freq)/1000);
  else
    snprintf(string,size,"%.3f "STRING_MEGAHERZ, (float)gpu->freq);

  return string;
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
  char* str;

  if(gpu->peak_performance == -1) {
    str = (char *) emalloc(sizeof(char) * (strlen(STRING_UNKNOWN) + 1));
    strncpy(str, STRING_UNKNOWN, strlen(STRING_UNKNOWN) + 1);
    return str;
  }

  // 7 for digits (e.g, XXXX.XX), 7 for XFLOP/s
  double flopsd = (double) gpu->peak_performance;
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
