#include <stdio.h>

#include "pci.hpp"
#include "chips.hpp"
#include "../common/global.hpp"
#include "../common/pci.hpp"

#define CHECK_PCI_START if (false) {}
#define CHECK_PCI(pci, id, chip) \
   else if (pci->device_id == id) return chip;
#define CHECK_PCI_END else { printBug("TODOO"); return CHIP_UNKNOWN_INTEL; }

GPUCHIP get_chip_from_pci(struct pci* pci) {
  CHECK_PCI_START
  CHECK_PCI(pci, 0x5917, CHIP_UHDG_620)
  CHECK_PCI_END
}
