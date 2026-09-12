#include "global.hpp"
#include "cpuid.hpp"
#include <cstdlib>
#include <cstring>
#include <cstdio>

#define CPU_VENDOR_MAX_LENGTH   13
#define CPU_NAME_MAX_LENGTH     49

void cpuid(uint32_t level, uint32_t *eax, uint32_t *ebx, uint32_t *ecx, uint32_t *edx) {
        __asm volatile("cpuid"
            : "=a" (*eax),
              "=b" (*ebx),
              "=c" (*ecx),
              "=d" (*edx)
            : "0" (level), "2" (*ecx));
}

char* get_cpu_vendor() {
  uint32_t vendor[3];
  uint32_t dummy;
  char * name = (char *) emalloc(sizeof(char) * CPU_VENDOR_MAX_LENGTH);
  memset(name, 0, CPU_VENDOR_MAX_LENGTH);

  cpuid(0x00000000, &dummy, vendor+0x0, vendor+0x2, vendor+0x1);

  snprintf(name, CPU_VENDOR_MAX_LENGTH, "%s", (char *) vendor);

  return name;
}

char* get_str_cpu_name_internal() {
  uint32_t brand[12];
  char * name = (char *) emalloc(sizeof(char) * CPU_NAME_MAX_LENGTH);
  memset(name, 0, CPU_NAME_MAX_LENGTH);

  cpuid(0x80000002, brand+0x0, brand+0x1, brand+0x2, brand+0x3);
  cpuid(0x80000003, brand+0x4, brand+0x5, brand+0x6, brand+0x7);
  cpuid(0x80000004, brand+0x8, brand+0x9, brand+0xa, brand+0xb);

  snprintf(name, CPU_NAME_MAX_LENGTH, "%s", (char *) brand);

  return name;
}

struct cpuid* get_cpuid() {
  uint32_t eax = 0x00000001;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;

  cpuid(0x00000001, &eax, &ebx, &ecx, &edx);

  struct cpuid* cpuid = (struct cpuid*) emalloc(sizeof(struct cpuid));
  cpuid->stepping = eax & 0xF;
  cpuid->model = (eax >> 4) & 0xF;
  cpuid->emodel = (eax >> 16) & 0xF;
  cpuid->family = (eax >> 8) & 0xF;
  cpuid->efamily = (eax >> 20) & 0xFF;

  return cpuid;
}