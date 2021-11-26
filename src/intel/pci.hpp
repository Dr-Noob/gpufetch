#ifndef __PCI_INTEL__
#define __PCI_INTEL__

#include <stdint.h>

#include "../common/pci.hpp"
#include "chips.hpp"

/*
 * doc: https://wiki.osdev.org/PCI#Class_Codes
 *      https://pci-ids.ucw.cz/read/PC
 */
#define PCI_VENDOR_ID_INTEL 0x8086

struct pci;

struct pci* get_pci_from_pciutils(struct pci_dev *devices);
GPUCHIP get_chip_from_pci(struct pci* pci);

#endif
