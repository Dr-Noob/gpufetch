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
  // Gen6
  CHECK_PCI(pci, 0x0102, CHIP_HD_2000)
  CHECK_PCI(pci, 0x0106, CHIP_HD_2000)
  CHECK_PCI(pci, 0x010A, CHIP_HD_2000)
  CHECK_PCI(pci, 0x0112, CHIP_HD_3000)
  CHECK_PCI(pci, 0x0122, CHIP_HD_3000)
  CHECK_PCI(pci, 0x0116, CHIP_HD_3000)
  CHECK_PCI(pci, 0x0126, CHIP_HD_3000)
  // Gen7
  CHECK_PCI(pci, 0x0152, CHIP_HD_2500)
  CHECK_PCI(pci, 0x0156, CHIP_HD_2500)
  CHECK_PCI(pci, 0x0162, CHIP_HD_4000)
  CHECK_PCI(pci, 0x0166, CHIP_HD_4000)
  CHECK_PCI(pci, 0x016a, CHIP_HD_P4000)
  // Gen7.5
  CHECK_PCI(pci, 0x0A1E, CHIP_HD_4200)
  CHECK_PCI(pci, 0x041E, CHIP_HD_4400)
  CHECK_PCI(pci, 0x0A16, CHIP_HD_4400)
  CHECK_PCI(pci, 0x0412, CHIP_HD_4600)
  CHECK_PCI(pci, 0x0416, CHIP_HD_4600)
  CHECK_PCI(pci, 0x0D12, CHIP_HD_4600)
  CHECK_PCI(pci, 0x041A, CHIP_HD_P4600)
  CHECK_PCI(pci, 0x0A2E, CHIP_IRIS_5100)
  CHECK_PCI(pci, 0x0D22, CHIP_IRISP_5200)
  CHECK_PCI(pci, 0x0D26, CHIP_IRISP_P5200)
  // Gen8
  CHECK_PCI(pci, 0x161E, CHIP_HD_5300)
  CHECK_PCI(pci, 0x1616, CHIP_HD_5500)
  CHECK_PCI(pci, 0x1612, CHIP_HD_5600)
  CHECK_PCI(pci, 0x161A, CHIP_HD_P5700)
  CHECK_PCI(pci, 0x1626, CHIP_HD_6000)
  CHECK_PCI(pci, 0x162B, CHIP_IRIS_6100)
  CHECK_PCI(pci, 0x1622, CHIP_IRISP_6200)
  CHECK_PCI(pci, 0x162A, CHIP_IRISP_P6300)
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
