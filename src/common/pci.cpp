#include "global.hpp"
#include "pci.hpp"
#include <cstddef>

#define CLASS_VGA_CONTROLLER 0x0300

uint16_t pciutils_get_pci_vendor_id(struct pci_dev *devices, int id) {
  for(struct pci_dev *dev=devices; dev != NULL; dev=dev->next) {
    if(dev->vendor_id == id && dev->device_class == CLASS_VGA_CONTROLLER) {
      return dev->vendor_id;
    }
  }

  printErr("Unable to find a valid device for id %d using pciutils", id);
  return 0;
}

uint16_t pciutils_get_pci_device_id(struct pci_dev *devices, int id) {
  for(struct pci_dev *dev=devices; dev != NULL; dev=dev->next) {
   if(dev->vendor_id == id && dev->device_class == CLASS_VGA_CONTROLLER) {
      return dev->device_id;
    }
  }

  printErr("Unable to find a valid device for id %d using pciutils", id);
  return 0;
}

struct pci* get_pci_from_pciutils(struct pci_dev *devices, int id) {
  struct pci* pci = (struct pci*) emalloc(sizeof(struct pci));

  pci->vendor_id = pciutils_get_pci_vendor_id(devices, id);
  pci->device_id = pciutils_get_pci_device_id(devices, id);

  return pci;
}

struct pci_dev *get_pci_devices_from_pciutils() {
  struct pci_access *pacc;
  struct pci_dev *dev;

  pacc = pci_alloc();
  pci_init(pacc);
  pci_scan_bus(pacc);

  for (dev=pacc->devices; dev; dev=dev->next) {
    pci_fill_info(dev, PCI_FILL_IDENT | PCI_FILL_BASES | PCI_FILL_CLASS);
  }

  return pacc->devices;
}
