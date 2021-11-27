#include <stdio.h>

#include "pci.hpp"
#include "chips.hpp"
#include "../common/global.hpp"
#include "../common/pci.hpp"

#define CHECK_PCI_START if (false) {}
#define CHECK_PCI(pci, id, chip) \
   else if (pci->device_id == id) return chip;
#define CHECK_PCI_END else { printBug("TODOO"); return CHIP_UNKNOWN_INTEL; }

/*
 * https://github.com/mesa3d/mesa/blob/main/include/pci_ids/i965_pci_ids.h
 */
GPUCHIP get_chip_from_pci(struct pci* pci) {
  CHECK_PCI_START
  // Gen9
  CHECK_PCI(pci, 0x1902, CHIP_HD_510)
  CHECK_PCI(pci, 0x1906, CHIP_HD_510)
  CHECK_PCI(pci, 0x190B, CHIP_HD_510)
  CHECK_PCI(pci, 0x191E, CHIP_HD_515)
  CHECK_PCI(pci, 0x1916, CHIP_HD_520)
  CHECK_PCI(pci, 0x1921, CHIP_HD_520)
  CHECK_PCI(pci, 0x1912, CHIP_HD_530)
  CHECK_PCI(pci, 0x191B, CHIP_HD_530)
  CHECK_PCI(pci, 0x191D, CHIP_HD_P530)
  /*CHECK_PCI(pci, 0x5917, CHIP_HD_540)
  CHECK_PCI(pci, 0x5917, CHIP_HD_550)
  CHECK_PCI(pci, 0x5917, CHIP_HD_P555)
  CHECK_PCI(pci, 0x5917, CHIP_HD_580)
  CHECK_PCI(pci, 0x5917, CHIP_HD_P580)*/
  // Gen9.5
  CHECK_PCI(pci, 0x3185, CHIP_UHD_600)
  CHECK_PCI(pci, 0x3184, CHIP_UHD_605)
  CHECK_PCI(pci, 0x5917, CHIP_UHD_620)
  CHECK_PCI(pci, 0x3E91, CHIP_UHD_630)
  CHECK_PCI(pci, 0x3E92, CHIP_UHD_630)
  CHECK_PCI(pci, 0x3E98, CHIP_UHD_630)
  CHECK_PCI(pci, 0x3E9B, CHIP_UHD_630)
  CHECK_PCI(pci, 0x9BC5, CHIP_UHD_630)
  CHECK_PCI(pci, 0x9BC8, CHIP_UHD_630)
  CHECK_PCI(pci, 0x5902, CHIP_HD_610)
  CHECK_PCI(pci, 0x5906, CHIP_HD_610)
  CHECK_PCI(pci, 0x590B, CHIP_HD_610)
  CHECK_PCI(pci, 0x591E, CHIP_HD_615)
  CHECK_PCI(pci, 0x5912, CHIP_HD_630)
  CHECK_PCI(pci, 0x591B, CHIP_HD_630)
  CHECK_PCI(pci, 0x591A, CHIP_HD_P630)
  CHECK_PCI(pci, 0x591D, CHIP_HD_P630)
  CHECK_PCI(pci, 0x5926, CHIP_IRISP_640)
  CHECK_PCI(pci, 0x5927, CHIP_IRISP_650)
  CHECK_PCI_END
}
