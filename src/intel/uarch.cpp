#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>

#include "../common/uarch.hpp"
#include "../common/global.hpp"
#include "../common/gpu.hpp"
#include "chips.hpp"
#include "pci.hpp"
#include "cpuid.hpp"

// Data not available
#define NA                   -1

// Unknown manufacturing process
#define UNK                  -1

/*
 * Mapping between iGPU and CPU uarchs
 * -----------------------------------
 * Gen6:   Sandy Bridge      (2th Gen)
 * Gen7:   Ivy Brdige        (3th Gen)
 * Gen7.5: Haswell           (4th Gen)
 * Gen8:   Broadwell         (5th Gen)
 * Gen9:   Skylake           (6th Gen)
 * Gen9.5: Kaby Lake
 * Gen11:  Ice Lake          (10th Gen)
 * Gen12:  Rocket/Tiger Lake (11th Gen)
 * Gen12:  Alder Lake        (12th Gen)
 */
enum {
  UARCH_UNKNOWN,
  UARCH_GEN6,
  UARCH_GEN7,
  UARCH_GEN7_5,
  UARCH_GEN8,
  UARCH_GEN9,
  UARCH_GEN9_5,
  UARCH_GEN11,
  UARCH_GEN12_RKL,
  UARCH_GEN12_TGL,
  UARCH_GEN12_ALD,
};

static const char *uarch_str[] = {
  /*[ARCH_UNKNOWN    = */ STRING_UNKNOWN,
  /*[ARCH_GEN6]      = */ "Gen6",
  /*[ARCH_GEN7]      = */ "Gen7",
  /*[ARCH_GEN7_5]    = */ "Gen7.5",
  /*[ARCH_GEN8]      = */ "Gen8",
  /*[ARCH_GEN9]      = */ "Gen9",
  /*[ARCH_GEN9_5]    = */ "Gen9.5",
  /*[ARCH_GEN11]     = */ "Gen11",
  /*[ARCH_GEN12_RKL] = */ "Xe",
  /*[ARCH_GEN12_TGL] = */ "Xe",
  /*[ARCH_GEN12_ALD] = */ "Xe",
};

// Graphic Tiers (GT)
enum {
  GT_UNKNOWN,
  GT0_5, // Saw that 0.5 thing in iris_pci_ids.h
  GT1,
  GT1_4, // GT1 with 4 EUs
  GT1_5,
  GT2,
  GT3,
  GT3e,
  GT4e
};

static const char *gt_str[] = {
  /*[GT_UNKNOWN] = */ STRING_UNKNOWN,
  /*[GT0_5]      = */ "GT0.5",
  /*[GT1]        = */ "GT1",
  /*[GT1_4]      = */ "GT1",
  /*[GT1_5]      = */ "GT1.5",
  /*[GT2]        = */ "GT2",
  /*[GT3]        = */ "GT3",
  /*[GT3e]       = */ "GT3e",
  /*[GT4e]       = */ "GT4e",
};

#define CHECK_UARCH_START if (false) {}
#define CHECK_UARCH(arch, chip_, str, uarch, gt, process) \
   else if (arch->chip == chip_) fill_uarch(arch, str, uarch, gt, process);
#define CHECK_UARCH_END else { printBug("map_chip_to_uarch_intel: Unknown chip id: %d", arch->chip); fill_uarch(arch, STRING_UNKNOWN, UARCH_UNKNOWN, GT_UNKNOWN, 0); }

#define CHECK_TOPO_START if (false) {}
#define CHECK_TOPO(topo, arch, uarch_, gt_, eu_sub, sub, sli) \
  else if(arch->uarch == uarch_ && arch->gt == gt_) fill_topo(topo, eu_sub, sub, sli);
#define CHECK_TOPO_CHIP(topo, arch, uarch_, chip_, eu_sub, sub, sli) \
  else if(arch->uarch == uarch_ && arch->chip == chip_) fill_topo(topo, eu_sub, sub, sli);
#define CHECK_TOPO_END else { printBug("get_topology_info: Invalid uarch and gt combination: '%s' and '%s'", arch->chip_str, get_str_gt(arch)); fill_topo(topo, UNK, UNK, UNK); }

void fill_topo(struct topology_i* topo_i, int32_t eu_sub, int32_t sub, int32_t sli) {
  topo_i->slices = sli;
  topo_i->subslices = sub;
  topo_i->eu_subslice = eu_sub;
}

void fill_uarch(struct uarch* arch, char const *str, MICROARCH u, int32_t gt, uint32_t process) {
  arch->chip_str = (char *) emalloc(sizeof(char) * (strlen(str)+1));
  strcpy(arch->chip_str, str);
  arch->uarch = u;
  arch->process = process;
  arch->gt = gt;
}

void map_chip_to_uarch_intel(struct uarch* arch) {
  CHECK_UARCH_START
  // Gen6
  CHECK_UARCH(arch, CHIP_HD_SANDY,     "HD Graphics (Sandy Bridge)", UARCH_GEN6,   GT1,   32)
  CHECK_UARCH(arch, CHIP_HD_2000,      "HD Graphics 2000",           UARCH_GEN6,   GT1,   32)
  CHECK_UARCH(arch, CHIP_HD_3000,      "HD Graphics 3000",           UARCH_GEN6,   GT2,   32)
  // Gen7
  CHECK_UARCH(arch, CHIP_HD_IVY,       "HD Graphics (Ivy Bridge)",   UARCH_GEN7,   GT1,   22)
  CHECK_UARCH(arch, CHIP_HD_SILVER,    "HD Graphics (Silvermont)",   UARCH_GEN7,   GT1_4, 22)
  CHECK_UARCH(arch, CHIP_HD_2500,      "HD Graphics 2500",           UARCH_GEN7,   GT1,   22)
  CHECK_UARCH(arch, CHIP_HD_4000,      "HD Graphics 4000",           UARCH_GEN7,   GT2,   22)
  CHECK_UARCH(arch, CHIP_HD_P4000,     "HD Graphics P4000",          UARCH_GEN7,   GT2,   22)
  // Gen7.5
  CHECK_UARCH(arch, CHIP_HD_HASWELL,   "HD Graphics (Haswell)",      UARCH_GEN7_5, GT1,   22)
  CHECK_UARCH(arch, CHIP_HD_4200,      "HD Graphics 4200",           UARCH_GEN7_5, GT2,   22)
  CHECK_UARCH(arch, CHIP_HD_4400,      "HD Graphics 4400",           UARCH_GEN7_5, GT2,   22)
  CHECK_UARCH(arch, CHIP_HD_4600,      "HD Graphics 4600",           UARCH_GEN7_5, GT2,   22)
  CHECK_UARCH(arch, CHIP_HD_P4600,     "HD Graphics P4600",          UARCH_GEN7_5, GT2,   22)
  CHECK_UARCH(arch, CHIP_IRIS_5100,    "HD Iris 5100",               UARCH_GEN7_5, GT3,   22)
  CHECK_UARCH(arch, CHIP_IRISP_5200,   "HD Iris Pro 5200",           UARCH_GEN7_5, GT3,   22)
  CHECK_UARCH(arch, CHIP_IRISP_P5200,  "HD Iris Pro P5200",          UARCH_GEN7_5, GT3,   22)
  // Gen8
  CHECK_UARCH(arch, CHIP_HD_BROADWELL, "HD Graphics (Broadwell)",    UARCH_GEN8,   GT1,   14)
  CHECK_UARCH(arch, CHIP_HD_5300,      "HD Graphics 5300",           UARCH_GEN8,   GT2,   14)
  CHECK_UARCH(arch, CHIP_HD_5500,      "HD Graphics 5500",           UARCH_GEN8,   GT2,   14)
  CHECK_UARCH(arch, CHIP_HD_5600,      "HD Graphics 5600",           UARCH_GEN8,   GT2,   14)
  CHECK_UARCH(arch, CHIP_HD_P5700,     "HD Graphics P5700",          UARCH_GEN8,   GT2,   14)
  CHECK_UARCH(arch, CHIP_HD_6000,      "HD Graphics 6000",           UARCH_GEN8,   GT3,   14)
  CHECK_UARCH(arch, CHIP_IRIS_6100,    "Iris Graphics 6100",         UARCH_GEN8,   GT3,   14)
  CHECK_UARCH(arch, CHIP_IRISP_6200,   "Iris Pro Graphics 6200",     UARCH_GEN8,   GT3,   14)
  CHECK_UARCH(arch, CHIP_IRISP_P6300,  "Iris Pro Graphics P6300",    UARCH_GEN8,   GT3,   14)
  // Gen9
  CHECK_UARCH(arch, CHIP_HD_510,       "HD Graphics 510",            UARCH_GEN9,   GT1,   14)
  CHECK_UARCH(arch, CHIP_HD_515,       "HD Graphics 515",            UARCH_GEN9,   GT2,   14)
  CHECK_UARCH(arch, CHIP_HD_520,       "HD Graphics 520",            UARCH_GEN9,   GT2,   14)
  CHECK_UARCH(arch, CHIP_HD_530,       "HD Graphics 530",            UARCH_GEN9,   GT2,   14)
  CHECK_UARCH(arch, CHIP_HD_P530,      "HD Graphics P530",           UARCH_GEN9,   GT2,   14)
  // Gen9.5
  CHECK_UARCH(arch, CHIP_UHD_600,      "UHD Graphics 600",           UARCH_GEN9_5, GT1,   14)
  CHECK_UARCH(arch, CHIP_UHD_605,      "UHD Graphics 605",           UARCH_GEN9_5, GT1_5, 14)
  CHECK_UARCH(arch, CHIP_UHD_620,      "UHD Graphics 620",           UARCH_GEN9_5, GT2,   14)
  CHECK_UARCH(arch, CHIP_UHD_630,      "UHD Graphics 630",           UARCH_GEN9_5, GT2,   14)
  CHECK_UARCH(arch, CHIP_HD_610,       "HD Graphics 610",            UARCH_GEN9_5, GT1,   14)
  CHECK_UARCH(arch, CHIP_HD_615,       "HD Graphics 615",            UARCH_GEN9_5, GT2,   14)
  CHECK_UARCH(arch, CHIP_HD_630,       "HD Graphics 630",            UARCH_GEN9_5, GT2,   14)
  CHECK_UARCH(arch, CHIP_HD_P630,      "HD Graphics P630",           UARCH_GEN9_5, GT2,   14)
  CHECK_UARCH(arch, CHIP_IRISP_640,    "Iris Plus Graphics 640",     UARCH_GEN9_5, GT3e,  14)
  CHECK_UARCH(arch, CHIP_IRISP_640,    "Iris Plus Graphics 650",     UARCH_GEN9_5, GT3e,  14)
  // Gen11
  CHECK_UARCH(arch, CHIP_UHD_G1,       "UHD Graphics G1",            UARCH_GEN11,  GT1,   10)
  CHECK_UARCH(arch, CHIP_IRISP_G4,     "Iris Plus Graphics G4",      UARCH_GEN11,  GT1_5, 10)
  CHECK_UARCH(arch, CHIP_IRISP_G7,     "Iris Plus Graphics G7",      UARCH_GEN11,  GT2,   10)
  // Xe (Gen12)
  CHECK_UARCH(arch, CHIP_UHD_710,      "UHD Graphics 710",           UARCH_GEN12_ALD, GT1,   10)
  CHECK_UARCH(arch, CHIP_UHD_730_ALD,  "UHD Graphics 730",           UARCH_GEN12_ALD, GT1,   10)
  CHECK_UARCH(arch, CHIP_UHD_770,      "UHD Graphics 770",           UARCH_GEN12_ALD, GT1,   10)
  CHECK_UARCH(arch, CHIP_UHD_730_RKL,  "UHD Graphics 730",           UARCH_GEN12_RKL, GT1,   14)
  CHECK_UARCH(arch, CHIP_UHD_750,      "UHD Graphics 750",           UARCH_GEN12_RKL, GT1,   14)
  CHECK_UARCH(arch, CHIP_XE_G4,        "Iris Xe G4",                 UARCH_GEN12_TGL, GT2,   10)
  CHECK_UARCH(arch, CHIP_XE_G7,        "Iris Xe G7",                 UARCH_GEN12_TGL, GT2,   10)
  CHECK_UARCH_END
}

const char* get_str_uarch_intel(struct uarch* arch) {
  return uarch_str[arch->uarch];
}

const char* get_str_gt(struct uarch* arch) {
  return gt_str[arch->gt];
}

struct uarch* get_uarch_from_pci(struct pci* pci) {
  struct uarch* arch = (struct uarch*) emalloc(sizeof(struct uarch));

  arch->chip_str = NULL;
  arch->chip = get_chip_from_pci_intel(pci);
  if(arch->chip == CHIP_UNKNOWN_INTEL) {
    return NULL;
  }
  else {
    map_chip_to_uarch_intel(arch);
    return arch;
  }
}

char* get_name_from_uarch(struct uarch* arch) {
  char* name = (char *) emalloc(sizeof(char) * (strlen(arch->chip_str) + 6 + 1));
  sprintf(name, "%s", arch->chip_str);
  return name;
}

/*
 * Refs:
 * Gen6:     https://en.wikipedia.org/wiki/List_of_Intel_graphics_processing_units#Gen6
 * Gen7/7.5: https://en.wikipedia.org/wiki/List_of_Intel_graphics_processing_units#Gen7
             "The Compute Architecture of Intel Processor Graphics Gen7.5, v1.0"
 * Gen8:     https://en.wikipedia.org/wiki/List_of_Intel_graphics_processing_units#Gen8
             "The Compute Architecture of Intel Processor Graphics Gen8, v1.1"
 * Gen9:     https://en.wikichip.org/wiki/intel/microarchitectures/gen9#Configuration
             "The Compute Architecture of Intel Processor Graphics Gen9, v1.0"
 * Gen9.5:   https://en.wikichip.org/wiki/intel/microarchitectures/gen9.5#Configuration

 * Also:     https://www.techpowerup.com/gpu-specs/intel-rocket-lake-gt1.g993
             https://www.techpowerup.com/gpu-specs/?architecture=Generation%2012.1
             https://elixir.bootlin.com/linux/latest/source/include/drm/i915_pciids.h
 */
struct topology_i* get_topology_info(struct uarch* arch) {
  struct topology_i* topo = (struct topology_i*) emalloc(sizeof(struct topology_i));

  // Syntax: (EU per subslice, Subslices, Slices)
  CHECK_TOPO_START
  // Gen6
  CHECK_TOPO(topo, arch, UARCH_GEN6,   GT1,   6, 1, 1)
  CHECK_TOPO(topo, arch, UARCH_GEN6,   GT2,   6, 2, 1)
  // Gen7
  CHECK_TOPO(topo, arch, UARCH_GEN7,   GT1_4, 4, 1, 1)
  CHECK_TOPO(topo, arch, UARCH_GEN7,   GT1,   6, 1, 1)
  CHECK_TOPO(topo, arch, UARCH_GEN7,   GT2,   8, 2, 1)
  CHECK_TOPO(topo, arch, UARCH_GEN7,   GT3,   6, 1, 1)
  // Gen7.5
  CHECK_TOPO(topo, arch, UARCH_GEN7_5, GT1,  10, 1, 1)
  CHECK_TOPO(topo, arch, UARCH_GEN7_5, GT2,  10, 2, 1)
  CHECK_TOPO(topo, arch, UARCH_GEN7_5, GT3,  10, 4, 1)
  // Gen8
  CHECK_TOPO(topo, arch, UARCH_GEN8,   GT1,   6, 2, 1)
  CHECK_TOPO(topo, arch, UARCH_GEN8,   GT2,   8, 3, 1)
  CHECK_TOPO(topo, arch, UARCH_GEN8,   GT3,   8, 6, 2)
  // Gen9
  CHECK_TOPO(topo, arch, UARCH_GEN9,   GT1,   6, 2, 1)
  CHECK_TOPO(topo, arch, UARCH_GEN9,   GT2,   8, 3, 1)
  CHECK_TOPO(topo, arch, UARCH_GEN9,   GT3,   8, 6, 2)
  CHECK_TOPO(topo, arch, UARCH_GEN9,   GT4e,  8, 9, 3)
  // Gen9.5
  CHECK_TOPO(topo, arch, UARCH_GEN9_5, GT1,   6, 2, 1)
  CHECK_TOPO(topo, arch, UARCH_GEN9_5, GT1_5, 6, 3, 1)
  CHECK_TOPO(topo, arch, UARCH_GEN9_5, GT2,   8, 3, 1)
  CHECK_TOPO(topo, arch, UARCH_GEN9_5, GT3,   8, 6, 2)
  CHECK_TOPO(topo, arch, UARCH_GEN9_5, GT3e,  8, 6, 2) // Same as GT3, but has eDRAM cache
  // Gen11
  CHECK_TOPO(topo, arch, UARCH_GEN11,  GT1,   8, 4, 1)
  CHECK_TOPO(topo, arch, UARCH_GEN11,  GT1_5, 8, 6, 1)
  CHECK_TOPO(topo, arch, UARCH_GEN11,  GT2,   8, 8, 1)
  // Xe (Gen12)
  // NOTE: Instead of checking for uarch + graphics tier,
  // we have to check for uarch + exact chip
  CHECK_TOPO_CHIP(topo, arch, UARCH_GEN12_RKL, CHIP_UHD_730_RKL, 8, 3, 1)
  CHECK_TOPO_CHIP(topo, arch, UARCH_GEN12_RKL, CHIP_UHD_750,     8, 4, 1)
  CHECK_TOPO_CHIP(topo, arch, UARCH_GEN12_TGL, CHIP_XE_G4,       8, 6, 1)
  else if(arch->uarch == UARCH_GEN12_TGL && arch->chip == CHIP_XE_G7) {
    // Special case: TigerLake GT2 needs to check if is i5/i7 to know the exact topology
    if(is_corei5()) {
      fill_topo(topo, 10, 8, 1); // Should be 80 EUs, but not sure about the organization
    }
    else {
      fill_topo(topo, 16, 6, 1);
    }
  }
  CHECK_TOPO_CHIP(topo, arch, UARCH_GEN12_ALD, CHIP_UHD_710,     8, 2, 1)
  CHECK_TOPO_CHIP(topo, arch, UARCH_GEN12_ALD, CHIP_UHD_730_ALD, 8, 3, 1)
  CHECK_TOPO_CHIP(topo, arch, UARCH_GEN12_ALD, CHIP_UHD_770,     8, 4, 1)
  // TODO: Add ALD UHD Graphics/Xe Graphics
  CHECK_TOPO_END
  return topo;
}
