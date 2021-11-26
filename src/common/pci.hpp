#ifndef __GPUFETCH_PCI__
#define __GPUFETCH_PCI__

#include <cstdint>
extern "C" {
  #include <pci/pci.h>
}

struct pci {
  uint16_t vendor_id;
  uint16_t device_id;
};

struct pci* get_pci_from_pciutils(struct pci_dev *devices, int id);
struct pci_dev *get_pci_devices_from_pciutils();

#endif
