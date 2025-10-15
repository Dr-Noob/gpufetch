#include <cstdlib>
#include <cstdint>
#include <cstring>

#include "../common/uarch.hpp"
#include "../common/global.hpp"
#include "../common/gpu.hpp"
#include "chips.hpp"

// MICROARCH values
enum {
  UARCH_UNKNOWN,
  // GCN (Graphics Core Next)
  // Empty for now
  // ...
  // RDNA (Radeon DNA)
  UARCH_RDNA,
  UARCH_RDNA2,
  UARCH_RDNA3,
  UARCH_RDNA4,
  // CDNA (Compute DNA)
  UARCH_CDNA,
  UARCH_CDNA2,
  UARCH_CDNA3,
  UARCH_CDNA4
};

static const char *uarch_str[] = {
  /*[ARCH_UNKNOWN]    = */ STRING_UNKNOWN,
  /*[UARCH_RDNA]      = */ "RDNA",
  /*[UARCH_RDNA2]     = */ "RDNA2",
  /*[UARCH_RDNA3]     = */ "RDNA3",
  /*[UARCH_RDNA4]     = */ "RDNA4",
  /*[UARCH_CDNA]      = */ "CDNA",
  /*[UARCH_CDNA2]     = */ "CDNA2",
  /*[UARCH_CDNA3]     = */ "CDNA3",
  /*[UARCH_CDNA4]     = */ "CDNA4",
};

// Sources: 
// - https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
// - https://www.techpowerup.com
//
// This is sometimes refered to as LLVM target, but also shader ISA.
//
// LLVM target *usually* maps to a specific architecture. However there
// are case where this is not true:
// MI8 is GCN3.0 with LLVM target gfx803
// MI6 is GCN4.0 with LLVM target gfx803
// or
// Strix Point can be gfx1150 or gfx1151
//
// NOTE: GCN chips are stored for completeness, but they are
// not actively supported.
enum {
  TARGET_UNKNOWN_HSA,
  /// GCN (Graphics Core Next)
  /// ------------------------
  // GCN 1.0 
  TARGET_GFX600,
  TARGET_GFX601,
  TARGET_GFX602,
  // GCN 2.0
  TARGET_GFX700,
  TARGET_GFX701,
  TARGET_GFX702,
  TARGET_GFX703,
  TARGET_GFX704,
  TARGET_GFX705,
  // GCN 3.0 / 4.0
  TARGET_GFX801,
  TARGET_GFX802,
  TARGET_GFX803,
  TARGET_GFX805,
  TARGET_GFX810,
  // GCN 5.0
  TARGET_GFX900,
  TARGET_GFX902,
  TARGET_GFX904,
  // GCN 5.1
  TARGET_GFX906,
  // ???
  TARGET_GFX909,
  TARGET_GFX90C,
  /// RDNA (Radeon DNA)
  /// -----------------
  // RDNA1
  TARGET_GFX1010,
  TARGET_GFX1011,
  TARGET_GFX1012,
  // RDNA2
  TARGET_GFX1013, // Oberon
  TARGET_GFX1030,
  TARGET_GFX1031,
  TARGET_GFX1032,
  TARGET_GFX1033,
  TARGET_GFX1034,
  TARGET_GFX1035, // ??
  TARGET_GFX1036, // ??
  // RDNA3
  TARGET_GFX1100,
  TARGET_GFX1101,
  TARGET_GFX1102,
  TARGET_GFX1103, // ???
  // RDNA3.5
  TARGET_GFX1150, // Strix Point
  TARGET_GFX1151, // Strix Halo / Strix Point
  TARGET_GFX1152, // Krackan Point
  TARGET_GFX1153, // ???
  // RDNA4
  TARGET_GFX1200,
  TARGET_GFX1201,
  TARGET_GFX1250, // ???
  TARGET_GFX1251, // ???
  /// CDNA (Compute DNA)
  /// ------------------
  // CDNA
  TARGET_GFX908,
  // CDNA2
  TARGET_GFX90A,
  // CDNA3
  TARGET_GFX942,
  // CDNA4
  TARGET_GFX950  
};

#define CHECK_UARCH_START if (false) {}
#define CHECK_UARCH(arch, chip_, str, uarch, process) \
   else if (arch->chip == chip_) fill_uarch(arch, str, uarch, process);
#define CHECK_UARCH_END else { if(arch->chip != CHIP_UNKNOWN_CUDA) printBug("map_chip_to_uarch_hsa: Unknown chip id: %d", arch->chip); fill_uarch(arch, STRING_UNKNOWN, UARCH_UNKNOWN, UNK); }

void fill_uarch(struct uarch* arch, char const *str, MICROARCH u, uint32_t process) {
  arch->chip_str = (char *) emalloc(sizeof(char) * (strlen(str)+1));
  strcpy(arch->chip_str, str);
  arch->uarch = u;
  arch->process = process;
}

// On chiplet based chips (such as Navi31, Navi32, etc),
// we have 2 different processes: The MCD process and the
// rest of the chip process. They might be different and here
// we just take one - let's take MCD process for now.
//
// TODO: Should we differentiate?
void map_chip_to_uarch_hsa(struct uarch* arch) {
  CHECK_UARCH_START

  // RDNA
  CHECK_UARCH(arch, CHIP_NAVI_10,  "Navi 10", UARCH_RDNA,  7)
  CHECK_UARCH(arch, CHIP_NAVI_12,  "Navi 12", UARCH_RDNA,  7)
  CHECK_UARCH(arch, CHIP_NAVI_14,  "Navi 14", UARCH_RDNA,  7)
  CHECK_UARCH(arch, CHIP_NAVI_21,  "Navi 21", UARCH_RDNA2, 7)
  CHECK_UARCH(arch, CHIP_NAVI_22,  "Navi 22", UARCH_RDNA2, 7)
  CHECK_UARCH(arch, CHIP_NAVI_23,  "Navi 23", UARCH_RDNA2, 7)
  CHECK_UARCH(arch, CHIP_NAVI_24,  "Navi 24", UARCH_RDNA2, 6)
  CHECK_UARCH(arch, CHIP_NAVI_31,  "Navi 31", UARCH_RDNA3, 6)
  CHECK_UARCH(arch, CHIP_NAVI_32,  "Navi 32", UARCH_RDNA3, 6)
  CHECK_UARCH(arch, CHIP_NAVI_33,  "Navi 33", UARCH_RDNA3, 6)
  CHECK_UARCH(arch, CHIP_NAVI_44,  "Navi 44", UARCH_RDNA4, 4)
  CHECK_UARCH(arch, CHIP_NAVI_48,  "Navi 48", UARCH_RDNA4, 4)
  // CDNA
  // NOTE: We will not show chip name for CDNA, thus use empty str
  CHECK_UARCH(arch, CHIP_ARCTURUS,        "", UARCH_CDNA,  7)
  CHECK_UARCH(arch, CHIP_ALDEBARAN,       "", UARCH_CDNA2, 6)
  CHECK_UARCH(arch, CHIP_AQUA_VANJARAM,   "", UARCH_CDNA3, 6)
  CHECK_UARCH(arch, CHIP_CDNA_NEXT,       "", UARCH_CDNA4, 6) // big difference between MCD and rest of the chip process
  
  CHECK_UARCH_END
}

#define CHECK_TGT_START if (false) {}
#define CHECK_TGT(target, llvm_target, chip) \
  else if (target == llvm_target) return chip;
#define CHECK_TGT_END else { printBug("LLVM target '%d' has no matching chip", target); return CHIP_UNKNOWN_HSA; }

// We have at least 2 choices to infer the chip:
//
// - LLVM target (e.g., gfx1101 is Navi 32)
// - PCI ID (e.g., 0x7470 is Navi 32)
//
// For now we will use the first approach, which seems to have
// some issues like mentioned in the enum.
// However PCI detection is also not perfect, since it is
// quite hard to find PCI ids from old hardware.
GPUCHIP get_chip_from_target_hsa(int32_t target) {
  CHECK_TGT_START
  /// RDNA
  /// -------------------------------------------
  CHECK_TGT(target, TARGET_GFX1010, CHIP_NAVI_10)
  CHECK_TGT(target, TARGET_GFX1011, CHIP_NAVI_12)
  CHECK_TGT(target, TARGET_GFX1012, CHIP_NAVI_14)
  // CHECK_TGT(target, TARGET_GFX1013, TODO)
  /// RDNA2
  /// -------------------------------------------
  CHECK_TGT(target, TARGET_GFX1030, CHIP_NAVI_21)
  CHECK_TGT(target, TARGET_GFX1031, CHIP_NAVI_22)
  CHECK_TGT(target, TARGET_GFX1032, CHIP_NAVI_23)
  CHECK_TGT(target, TARGET_GFX1033, CHIP_NAVI_21)
  CHECK_TGT(target, TARGET_GFX1034, CHIP_NAVI_24)
  // CHECK_TGT(target, TARGET_GFX1035, TODO)
  // CHECK_TGT(target, TARGET_GFX1036, TODO)
  /// RDNA3
  /// -------------------------------------------
  CHECK_TGT(target, TARGET_GFX1100, CHIP_NAVI_31)
  CHECK_TGT(target, TARGET_GFX1101, CHIP_NAVI_32)
  CHECK_TGT(target, TARGET_GFX1102, CHIP_NAVI_33)
  // CHECK_TGT(target, TARGET_GFX1103, TODO)
  /// RDNA3.5
  /// -------------------------------------------
  // CHECK_TGT(target, TARGET_GFX1150, TODO)
  // CHECK_TGT(target, TARGET_GFX1151, TODO)
  // CHECK_TGT(target, TARGET_GFX1152, TODO)
  // CHECK_TGT(target, TARGET_GFX1153, TODO)
  /// RDNA4
  /// -------------------------------------------
  CHECK_TGT(target, TARGET_GFX1200, CHIP_NAVI_44)
  CHECK_TGT(target, TARGET_GFX1201, CHIP_NAVI_48)
  // CHECK_TGT(target, TARGET_GFX1250, TODO)
  // CHECK_TGT(target, TARGET_GFX1251, TODO)
  /// CDNA
  /// -------------------------------------------
  CHECK_TGT(target, TARGET_GFX908, CHIP_ARCTURUS)
  /// CDNA2
  /// -------------------------------------------
  CHECK_TGT(target, TARGET_GFX90A, CHIP_ALDEBARAN)
  /// CDNA3
  /// -------------------------------------------
  CHECK_TGT(target, TARGET_GFX942, CHIP_AQUA_VANJARAM)
  /// CDNA4
  /// -------------------------------------------
  CHECK_TGT(target, TARGET_GFX950, CHIP_CDNA_NEXT)
  CHECK_TGT_END
}

#define CHECK_TGT_STR_START if (false) {}
#define CHECK_TGT_STR(target, llvm_target, chip) \
  else if (strcmp(target, llvm_target) == 0) return chip;
#define CHECK_TGT_STR_END else { return TARGET_UNKNOWN_HSA; }

// Maps the LLVM target string to the enum value
int32_t get_llvm_target_from_str(char* target) {
  // TODO: Autogenerate this
  // TODO: Add all, not only the ones we support in get_chip_from_target_hsa
  CHECK_TGT_STR_START
  CHECK_TGT_STR(target, "gfx1010", TARGET_GFX1010)
  CHECK_TGT_STR(target, "gfx1011", TARGET_GFX1011)
  CHECK_TGT_STR(target, "gfx1012", TARGET_GFX1012)
  CHECK_TGT_STR(target, "gfx1013", TARGET_GFX1013)
  CHECK_TGT_STR(target, "gfx1030", TARGET_GFX1030)
  CHECK_TGT_STR(target, "gfx1031", TARGET_GFX1031)
  CHECK_TGT_STR(target, "gfx1032", TARGET_GFX1032)
  CHECK_TGT_STR(target, "gfx1033", TARGET_GFX1033)
  CHECK_TGT_STR(target, "gfx1034", TARGET_GFX1034)
  CHECK_TGT_STR(target, "gfx1035", TARGET_GFX1035)
  CHECK_TGT_STR(target, "gfx1036", TARGET_GFX1036)
  CHECK_TGT_STR(target, "gfx1100", TARGET_GFX1100)
  CHECK_TGT_STR(target, "gfx1101", TARGET_GFX1101)
  CHECK_TGT_STR(target, "gfx1102", TARGET_GFX1102)
  CHECK_TGT_STR(target, "gfx1103", TARGET_GFX1103)
  CHECK_TGT_STR(target, "gfx1200", TARGET_GFX1200)
  CHECK_TGT_STR(target, "gfx1201", TARGET_GFX1201)
  CHECK_TGT_STR(target, "gfx1250", TARGET_GFX1250)
  CHECK_TGT_STR(target, "gfx1251", TARGET_GFX1251)
  CHECK_TGT_STR(target, "gfx908",  TARGET_GFX908)
  CHECK_TGT_STR(target, "gfx90a",  TARGET_GFX90A)
  CHECK_TGT_STR(target, "gfx942",  TARGET_GFX942)
  CHECK_TGT_STR(target, "gfx950",  TARGET_GFX950)
  CHECK_TGT_STR_END
}

struct uarch* get_uarch_from_hsa(struct gpu_info* gpu, char* gpu_name) {
  struct uarch* arch = (struct uarch*) emalloc(sizeof(struct uarch));

  arch->llvm_target = get_llvm_target_from_str(gpu_name);
  if (arch->llvm_target == TARGET_UNKNOWN_HSA) {
    printErr("Unknown LLVM target: '%s'", gpu_name);
    return NULL;
  }

  arch->chip_str = NULL;
  arch->chip = get_chip_from_target_hsa(arch->llvm_target);
  map_chip_to_uarch_hsa(arch);

  return arch;
}

bool is_uarch_valid(struct uarch* arch) {
  if (arch == NULL) {
    printBug("Invalid uarch: arch is NULL");
    return false;
  }
  if (arch->uarch >= UARCH_UNKNOWN && arch->uarch <= UARCH_CDNA4) {
    return true;
  }
  else {
    printBug("Invalid uarch: %d", arch->uarch);
    return false;
  }
}

const char* get_str_uarch_hsa(struct uarch* arch) {
  if (!is_uarch_valid(arch)) {
    return NULL;
  }
  return uarch_str[arch->uarch];
}