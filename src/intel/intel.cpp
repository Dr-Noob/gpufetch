#include <stdio.h>
#include <string.h>

#include "intel.hpp"
#include "uarch.hpp"
#include "chips.hpp"
#include "udev.hpp"
#include "../common/pci.hpp"
#include "../common/global.hpp"

int64_t get_peak_performance_intel(struct gpu_info* gpu) {
  return gpu->freq * 1000000 * gpu->topo_i->eu_subslice * gpu->topo_i->subslices * 8 * 2;
}

struct gpu_info* get_gpu_info_intel() {
  struct gpu_info* gpu = (struct gpu_info*) emalloc(sizeof(struct gpu_info));
  gpu->vendor = GPU_VENDOR_INTEL;

  struct pci_dev *devices = get_pci_devices_from_pciutils();
  gpu->pci = get_pci_from_pciutils(devices, PCI_VENDOR_ID_INTEL);

  if(gpu->pci == NULL) {
    // No Intel iGPU found in PCI, which means it is not present
    return NULL;
  }

  gpu->arch = get_uarch_from_pci(gpu->pci);
  gpu->name = get_name_from_uarch(gpu->arch);
  gpu->topo_i = get_topology_info(gpu->arch);
  gpu->freq = get_max_freq_from_file(gpu->pci);
  gpu->peak_performance = get_peak_performance_intel(gpu);

  return gpu;
}

bool print_gpu_intel(struct gpu_info* gpu) {
  if(gpu->vendor != GPU_VENDOR_INTEL) return false;

  printf("Intel %s\n", gpu->name);

  return true;
}

char* get_str_eu(struct gpu_info* gpu) {
  return get_str_generic(gpu->topo_i->subslices * gpu->topo_i->eu_subslice);
}
