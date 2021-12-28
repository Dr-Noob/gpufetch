#ifndef __GPUFETCH_PCI__
#define __GPUFETCH_PCI__

#include <cstdint>
extern "C" {
  #include <pci/pci.h>
}

struct pci {
  uint16_t vendor_id;
  uint16_t device_id;
  uint16_t domain;
  uint16_t bus;
  uint16_t dev;
  uint16_t func;
};

struct pci* get_pci_from_pciutils(struct pci_dev *devices, int id);
struct pci_dev *get_pci_devices_from_pciutils();
void print_gpus_list_pci();

#endif
