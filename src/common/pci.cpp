#include "global.hpp"
#include "pci.hpp"
#include "../cuda/pci.hpp"
#include "../intel/pci.hpp"

#include <cstdio>
#include <cstddef>

// TODO: Move AMD PCI id when possible
#define PCI_VENDOR_ID_AMD    0x1002
#define CLASS_VGA_CONTROLLER 0x0300

bool pciutils_is_vendor_id_present(struct pci_dev *devices, int id) {
  for(struct pci_dev *dev=devices; dev != NULL; dev=dev->next) {
    if(dev->vendor_id == id && dev->device_class == CLASS_VGA_CONTROLLER) {
      return true;
    }
  }

  printWarn("Unable to find a valid device for vendor id 0x%.4X using pciutils", id);
  return false;
}

uint16_t pciutils_get_pci_device_id(struct pci_dev *devices, int id) {
  for(struct pci_dev *dev=devices; dev != NULL; dev=dev->next) {
   if(dev->vendor_id == id && dev->device_class == CLASS_VGA_CONTROLLER) {
      return dev->device_id;
    }
  }

  printErr("Unable to find a valid device for device id 0x%.4X using pciutils", id);
  return 0;
}

void pciutils_set_pci_bus(struct pci* pci, struct pci_dev *devices, int id) {
  bool found = false;

  for(struct pci_dev *dev=devices; dev != NULL; dev=dev->next) {
   if(dev->vendor_id == id && dev->device_class == CLASS_VGA_CONTROLLER) {
      pci->domain = dev->domain;
      pci->bus = dev->bus;
      pci->dev = dev->dev;
      pci->func = dev->func;
      found = true;
    }
  }

  if(!found) printErr("Unable to find a valid device for id 0x%.4X using pciutils", id);
}

struct pci* get_pci_from_pciutils(struct pci_dev *devices, int id) {
  struct pci* pci = (struct pci*) emalloc(sizeof(struct pci));

  // TODO: Refactor this; instead of 2xGet + 1xSet, do it better
  if(pciutils_is_vendor_id_present(devices, id)) {
    pci->vendor_id = id;
    pci->device_id = pciutils_get_pci_device_id(devices, id);
    pciutils_set_pci_bus(pci, devices, id);
    return pci;
  }
  else {
    return NULL;
  }
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

void print_gpus_list_pci() {
  int i=0;
  struct pci_dev *devices = get_pci_devices_from_pciutils();

  for(struct pci_dev *dev=devices; dev != NULL; dev=dev->next) {
   if(dev->device_class == CLASS_VGA_CONTROLLER) {
      printf("- GPU %d: ", i);
      if(dev->vendor_id == PCI_VENDOR_ID_NVIDIA) {
        printf("NVIDIA ");
      }
      else if(dev->vendor_id == PCI_VENDOR_ID_INTEL) {
        printf("Intel ");
      }
      else if(dev->vendor_id == PCI_VENDOR_ID_AMD) {
        printf("AMD ");
      }
      printf("%.4x:%.4x\n", dev->vendor_id, dev->device_id);
    }
  }
}
