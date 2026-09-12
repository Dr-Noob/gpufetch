#ifndef __CPUID__
#define __CPUID__

#include <cstdint>

struct cpuid {
  uint32_t stepping;
  uint32_t model;
  uint32_t emodel;
  uint32_t family;
  uint32_t efamily;
};

void cpuid(uint32_t level, uint32_t *eax, uint32_t *ebx, uint32_t *ecx, uint32_t *edx);
char* get_cpu_vendor();
char* get_str_cpu_name_internal();
struct cpuid* get_cpuid();

#endif
