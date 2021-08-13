#ifndef __PCI__
#define __PCI__

#include <stdint.h>
#include "nvmlb.hpp"
#include "chips.hpp"

struct pci;

struct pci* get_pci_from_nvml(struct nvml_data* data);
GPUCHIP get_chip_from_pci(struct pci* pci);

#endif
