#include <stdint.h>
#include <cstddef>
#include <string.h>

#include "../common/uarch.hpp"
#include "../common/global.hpp"
#include "../common/gpu.hpp"
#include "chips.hpp"

// Data not available
#define NA                   -1

// Unknown manufacturing process
#define UNK                  -1

// MICROARCH values
enum {
  UARCH_UNKNOWN,
  UARCH_GEN9,
  UARCH_GEN9_5,
};

static const char *uarch_str[] = {
  /*[ARCH_UNKNOWN    = */ STRING_UNKNOWN,
  /*[ARCH_GEN9]      = */ "Gen9",
  /*[ARCH_GEN9.5]    = */ "Gen9.5",
};

#define CHECK_UARCH_START if (false) {}
#define CHECK_UARCH(arch, chip_, str, uarch, process) \
   else if (arch->chip == chip_) fill_uarch(arch, str, uarch, process);
#define CHECK_UARCH_END else { printBug("map_chip_to_uarch: Unknown chip id: %d", arch->chip); fill_uarch(arch, STRING_UNKNOWN, UARCH_UNKNOWN, 0); }

void fill_uarch(struct uarch* arch, char const *str, MICROARCH u, uint32_t process) {
  arch->chip_str = (char *) emalloc(sizeof(char) * (strlen(str)+1));
  strcpy(arch->chip_str, str);
  arch->uarch = u;
  arch->process = process;
}

void map_chip_to_uarch(struct uarch* arch) {
  CHECK_UARCH_START
  CHECK_UARCH(arch, CHIP_UHDG_620, "UHD Graphics 620", UARCH_GEN9_5, 14)
  CHECK_UARCH_END
}

struct uarch* get_uarch_from_pci(struct pci* pci) {
  struct uarch* arch = (struct uarch*) emalloc(sizeof(struct uarch));

  arch->chip_str = NULL;
  arch->chip = get_chip_from_pci(pci);
  map_chip_to_uarch(arch);

  return arch;
}

char* get_name_from_uarch(struct uarch* arch) {
  return arch->chip_str;
}
