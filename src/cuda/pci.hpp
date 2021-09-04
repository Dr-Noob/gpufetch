#ifndef __PCI_CUDA__
#define __PCI_CUDA__

#include <stdint.h>

#include "../common/pci.hpp"
#include "chips.hpp"

struct pci;

struct pci* get_pci_from_pciutils(struct pci_dev *devices);
GPUCHIP get_chip_from_pci(struct pci* pci);

#endif
