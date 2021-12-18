#ifndef __PCI_CUDA__
#define __PCI_CUDA__

#include <stdint.h>

#include "../common/pci.hpp"
#include "chips.hpp"

/*
 * doc: https://wiki.osdev.org/PCI#Class_Codes
 *      https://pci-ids.ucw.cz/read/PC
 */
#define PCI_VENDOR_ID_NVIDIA 0x10de

struct pci;

GPUCHIP get_chip_from_pci_cuda(struct pci* pci);

#endif
