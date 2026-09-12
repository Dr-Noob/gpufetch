#include "../common/global.hpp"
#include "../common/cpuid.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#define CPU_VENDOR_INTEL_STRING "GenuineIntel"

bool is_corei5() {
  uint32_t eax = 0;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;

  // Get CPU vendor
  char* cpu_vendor = get_cpu_vendor();

  if(strcmp(CPU_VENDOR_INTEL_STRING, cpu_vendor) != 0) {
    printBug("is_corei5: invalid CPU vendor: %s", cpu_vendor);
    return false;
  }

  cpuid(0x80000000, &eax, &ebx, &ecx, &edx);

  if (eax < 0x80000004){
    printBug("is_corei5: unexpected max extended level: 0x%.8X", eax);
    return false;
  }

  // Get CPU name
  char* cpu_name = get_str_cpu_name_internal();
  return strstr(cpu_name, "i5") != NULL;
}
