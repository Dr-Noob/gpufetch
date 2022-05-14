#include <cstdio>
#include <cstring>

#include "intel.hpp"
#include "uarch.hpp"
#include "chips.hpp"
#include "udev.hpp"
#include "../common/pci.hpp"
#include "../common/global.hpp"

int64_t get_peak_performance_intel(struct gpu_info* gpu) {
  // Check that we have valid data
  if(gpu->topo_i->eu_subslice < 0 ||
     gpu->topo_i->subslices < 0   ||
     gpu->freq <= 0)
  {
    return -1;
  }
  return gpu->freq * 1000000 * gpu->topo_i->eu_subslice * gpu->topo_i->subslices * 8 * 2;
}

struct gpu_info* get_gpu_info_intel(struct pci_dev *devices) {
  struct gpu_info* gpu = (struct gpu_info*) emalloc(sizeof(struct gpu_info));
  gpu->vendor = GPU_VENDOR_INTEL;
  gpu->pci = get_pci_from_pciutils(devices, PCI_VENDOR_ID_INTEL, 0);

  if(gpu->pci == NULL) {
    // No Intel iGPU found in PCI, which means it is not present
    return NULL;
  }

  gpu->arch = get_uarch_from_pci(gpu->pci);

  if(gpu->arch == NULL) {
    // No Intel iGPU found in arch LUT, which means it is not supported
    return NULL;
  }

  gpu->name = get_name_from_uarch(gpu->arch);
  gpu->topo_i = get_topology_info(gpu->arch);
  gpu->freq = get_max_freq_from_file(gpu->pci);
  gpu->peak_performance = get_peak_performance_intel(gpu);

  return gpu;
}

bool print_gpu_intel(struct gpu_info* gpu) {
  if(gpu->vendor != GPU_VENDOR_INTEL) return false;

  printf("%s\n", gpu->name);

  return true;
}

char* get_str_eu(struct gpu_info* gpu) {
  if(gpu->topo_i->subslices < 0 || gpu->topo_i->eu_subslice < 0)
    return get_str_generic(-1);
  return get_str_generic(gpu->topo_i->subslices * gpu->topo_i->eu_subslice);
}
