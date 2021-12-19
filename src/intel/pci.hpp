#ifndef __PCI_INTEL__
#define __PCI_INTEL__

#include <cstdint>

#include "../common/pci.hpp"
#include "chips.hpp"

/*
 * doc: https://wiki.osdev.org/PCI#Class_Codes
 *      https://pci-ids.ucw.cz/read/PC
 */
#define PCI_VENDOR_ID_INTEL 0x8086

struct pci;

GPUCHIP get_chip_from_pci_intel(struct pci* pci);

#endif
