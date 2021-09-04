#ifndef __GPUFETCH_PCI__
#define __GPUFETCH_PCI__

#include <cstdint>
extern "C" {
  #include <pci/pci.h>
}

uint16_t pciutils_get_pci_vendor_id(struct pci_dev *devices);
uint16_t pciutils_get_pci_device_id(struct pci_dev *devices);
struct pci_dev *get_pci_devices_from_pciutils();

#endif
