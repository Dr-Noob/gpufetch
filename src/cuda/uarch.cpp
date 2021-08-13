#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdint.h>
#include <cstddef>

#include "../common/global.hpp"
#include "../common/gpu.hpp"
#include "chips.hpp"

typedef uint32_t MICROARCH;

// Data not available
#define NA                   -1

// Unknown manufacturing process
#define UNK                  -1

// MICROARCH values
enum {
  UARCH_UNKNOWN,
  UARCH_TESLA,
  UARCH_FERMI,
  UARCH_KEPLER,
  UARCH_MAXWELL,
  UARCH_PASCAL,
  UARCH_VOLTA,
  UARCH_TURING,
  UARCH_AMPERE,
};

static const char *uarch_str[] = {
  /*[ARCH_UNKNOWN     = */ STRING_UNKNOWN,
  /*[ARCH_TESLA]      = */ "Tesla",
  /*[ARCH_FERMI]      = */ "Fermi",
  /*[ARCH_KEPLER]     = */ "Kepler",
  /*[ARCH_MAXWELL]    = */ "Maxwell",
  /*[ARCH_PASCAL]     = */ "Pascal",
  /*[ARCH_VOLTA]      = */ "Volta",
  /*[ARCH_TURING]     = */ "Turing",
  /*[ARCH_AMPERE]     = */ "Ampere",
};

struct uarch {
  int32_t cc_major;
  int32_t cc_minor;
  int32_t compute_capability;

  MICROARCH uarch;
  GPUCHIP chip;

  int32_t process;
  char* uarch_str;
  char* chip_str;
};

void map_cc_to_uarch(struct uarch* arch) {
  switch(arch->compute_capability) {
    case 10:
    case 11:
    case 12:
    case 13:
      arch->uarch = UARCH_TESLA;
      break;
    case 20:
    case 21:
      arch->uarch = UARCH_FERMI;
      break;
    case 30:
    case 32:
    case 35:
    case 37:
      arch->uarch = UARCH_KEPLER;
      break;
    case 50:
    case 52:
    case 53:
      arch->uarch = UARCH_MAXWELL;
      break;
    case 60:
    case 61:
    case 62:
      arch->uarch = UARCH_PASCAL;
      break;
    case 70:
    case 72:
      arch->uarch = UARCH_VOLTA;
      break;
    case 75:
      arch->uarch = UARCH_TURING;
      break;
    case 80:
    case 86:
      arch->uarch = UARCH_AMPERE;
      break;
    default:
      arch->uarch = UARCH_UNKNOWN;
      printErr("Invalid uarch: %d.%d\n", arch->cc_major, arch->cc_minor);
  }
}

struct uarch* get_uarch_from_cuda(struct gpu_info* gpu) {
  struct uarch* arch = (struct uarch*) emalloc(sizeof(struct uarch));

  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  arch->chip_str = NULL;
  arch->cc_major = deviceProp.major;
  arch->cc_minor = deviceProp.minor;
  arch->compute_capability = deviceProp.major * 10 + deviceProp.minor;

  map_cc_to_uarch(arch);

  arch->chip = get_chip_from_pci(gpu->pci);

  return arch;
}

const char* get_str_uarch(struct uarch* arch) {
  return uarch_str[arch->uarch];
}

char* get_str_cc(struct uarch* arch) {
  uint32_t max_size = 4;
  char* cc = (char *) ecalloc(max_size, sizeof(char));
  snprintf(cc, max_size, "%d.%d", arch->cc_major, arch->cc_minor);
  return cc;
}

char* get_str_process(struct uarch* arch) {
  return NULL;
}

char* get_str_chip(struct uarch* arch) {
  return arch->chip_str;
}

void free_uarch_struct(struct uarch* arch) {

}
