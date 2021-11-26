#include <stdio.h>
#include <string.h>

#include "intel.hpp"
#include "uarch.hpp"
#include "chips.hpp"
#include "../common/pci.hpp"
#include "../common/global.hpp"

struct gpu_info* get_gpu_info_intel() {
  struct gpu_info* gpu = (struct gpu_info*) emalloc(sizeof(struct gpu_info));
  const char* name = "UHD Graphics XXX";

  gpu->vendor = GPU_VENDOR_INTEL;

  struct pci_dev *devices = get_pci_devices_from_pciutils();
  gpu->pci = get_pci_from_pciutils(devices, PCI_VENDOR_ID_INTEL);
  gpu->arch = get_uarch_from_pci(gpu->pci);
  gpu->name = get_name_from_uarch(gpu->arch);

  return gpu;
}

bool print_gpu_intel(struct gpu_info* gpu) {
  if(gpu->vendor != GPU_VENDOR_INTEL) return false;

  printf("Intel %s\n", gpu->name);

  return true;
}
