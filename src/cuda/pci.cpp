#include <stdio.h>

#include "pci.hpp"
#include "nvmlb.hpp"
#include "../common/global.hpp"

struct pci {
  uint16_t vendor_id;
  uint16_t device_id;
};

struct pci* get_pci_from_nvml(struct nvml_data* data) {
  struct pci* pci = (struct pci*) emalloc(sizeof(struct pci));

  pci->vendor_id = nvml_get_pci_vendor_id(data);
  pci->device_id = nvml_get_pci_device_id(data);

  printf("pci->vendor_id=0x%.4X\n", pci->vendor_id);
  printf("pci->device_id=0x%.4X\n", pci->device_id);

  return pci;
}
