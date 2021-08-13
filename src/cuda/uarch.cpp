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

/*
 * - manufacturing process values were retrieved from techpowerup
 * - chip code names details:
 * o CHIP_XXXGL: indicates a professional-class (Quadro/Tesla) chip
 * o CHIP_XXXM:  indicates a mobile chip
 */
void map_chip_to_uarch(struct uarch* arch) {
  CHECK_UARCH_START
  // TESLA (1.0, 1.1, 1.2, 1.3)                                //
  CHECK_UARCH(arch, CHIP_G80,      "G80",      UARCH_TESLA,   90)
  CHECK_UARCH(arch, CHIP_G80GL,    "G80GL",    UARCH_TESLA,   90)
  CHECK_UARCH(arch, CHIP_G84,      "G84",      UARCH_TESLA,   80)
  CHECK_UARCH(arch, CHIP_G84GL,    "G84GL",    UARCH_TESLA,   80)
  CHECK_UARCH(arch, CHIP_G84GLM,   "G84GLM",   UARCH_TESLA,   80)
  CHECK_UARCH(arch, CHIP_G84M,     "G84M",     UARCH_TESLA,   80)
  CHECK_UARCH(arch, CHIP_G86,      "G86",      UARCH_TESLA,   80)
  CHECK_UARCH(arch, CHIP_G86GLM,   "G86GLM",   UARCH_TESLA,   80)
  CHECK_UARCH(arch, CHIP_G86M,     "G86M",     UARCH_TESLA,   80)
  CHECK_UARCH(arch, CHIP_G92,      "G92",      UARCH_TESLA,   65)
  CHECK_UARCH(arch, CHIP_G92GL,    "G92GL",    UARCH_TESLA,   65)
  CHECK_UARCH(arch, CHIP_G92GLM,   "G92GLM",   UARCH_TESLA,   65)
  CHECK_UARCH(arch, CHIP_G92M,     "G92M",     UARCH_TESLA,   65)
  CHECK_UARCH(arch, CHIP_G94,      "G94",      UARCH_TESLA,   65)
  CHECK_UARCH(arch, CHIP_G94GL,    "G94GL",    UARCH_TESLA,   65)
  CHECK_UARCH(arch, CHIP_G94GLM,   "G94GLM",   UARCH_TESLA,   65)
  CHECK_UARCH(arch, CHIP_G94M,     "G94M",     UARCH_TESLA,   65)
  CHECK_UARCH(arch, CHIP_G96,      "G96",      UARCH_TESLA,   65)
  CHECK_UARCH(arch, CHIP_G96C,     "G96C",     UARCH_TESLA,   55)
  CHECK_UARCH(arch, CHIP_G96CGL,   "G96CGL",   UARCH_TESLA,   55)
  CHECK_UARCH(arch, CHIP_G96CM,    "G96CM",    UARCH_TESLA,   55)
  CHECK_UARCH(arch, CHIP_G96GL,    "G96GL",    UARCH_TESLA,   65)
  CHECK_UARCH(arch, CHIP_G96GLM,   "G96GLM",   UARCH_TESLA,   65)
  CHECK_UARCH(arch, CHIP_G96M,     "G96M",     UARCH_TESLA,   65)
  CHECK_UARCH(arch, CHIP_G98,      "G98",      UARCH_TESLA,   65)
  CHECK_UARCH(arch, CHIP_G98GL,    "G98GL",    UARCH_TESLA,   65)
  CHECK_UARCH(arch, CHIP_G98GLM,   "G98GLM",   UARCH_TESLA,   65)
  CHECK_UARCH(arch, CHIP_G98M,     "G98M",     UARCH_TESLA,   65)
  CHECK_UARCH(arch, CHIP_GT200,    "GT200",    UARCH_TESLA,   65)
  CHECK_UARCH(arch, CHIP_GT200B,   "GT200B",   UARCH_TESLA,   55)
  CHECK_UARCH(arch, CHIP_GT200GL,  "GT200GL",  UARCH_TESLA,   55)
  CHECK_UARCH(arch, CHIP_GT215,    "GT215",    UARCH_TESLA,   40)
  CHECK_UARCH(arch, CHIP_GT215GLM, "GT215GLM", UARCH_TESLA,   40)
  CHECK_UARCH(arch, CHIP_GT215M,   "GT215M",   UARCH_TESLA,   40)
  CHECK_UARCH(arch, CHIP_GT216,    "GT216",    UARCH_TESLA,   40)
  CHECK_UARCH(arch, CHIP_GT216GL,  "GT216GL",  UARCH_TESLA,   40)
  CHECK_UARCH(arch, CHIP_GT216GLM, "GT216GLM", UARCH_TESLA,   40)
  CHECK_UARCH(arch, CHIP_GT216M,   "GT216M",   UARCH_TESLA,   40)
  CHECK_UARCH(arch, CHIP_GT218,    "GT218",    UARCH_TESLA,   40)
  CHECK_UARCH(arch, CHIP_GT218GL,  "GT218GL",  UARCH_TESLA,   40)
  CHECK_UARCH(arch, CHIP_GT218GLM, "GT218GLM", UARCH_TESLA,   40)
  CHECK_UARCH(arch, CHIP_GT218M,   "GT218M",   UARCH_TESLA,   40)
  // FERMI (2.0, 2.1)                                          //
  CHECK_UARCH(arch, CHIP_GF100,    "GF100",    UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF100GL,  "GF100GL",  UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF100GLM, "GF100GLM", UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF100M,   "GF100M",   UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF104,    "GF104",    UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF104GLM, "GF104GLM", UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF104M,   "GF104M",   UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF106,    "GF106",    UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF106GL,  "GF106GL",  UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF106GLM, "GF106GLM", UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF106M,   "GF106M",   UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF108,    "GF108",    UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF108GL,  "GF108GL",  UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF108GLM, "GF108GLM", UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF108M,   "GF108M",   UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF110,    "GF110",    UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF110GL,  "GF110GL",  UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF114,    "GF114",    UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF114M,   "GF114M",   UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF116,    "GF116",    UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF116M,   "GF116M",   UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF117M,   "GF117M",   UARCH_FERMI,   28)
  CHECK_UARCH(arch, CHIP_GF119,    "GF119",    UARCH_FERMI,   40)
  CHECK_UARCH(arch, CHIP_GF119M,   "GF119M",   UARCH_FERMI,   40)
  // KEPLER (3.0, 3.2, 3.5, 3.7                                //
  CHECK_UARCH(arch, CHIP_GK104,    "GK104",    UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK104GL,  "GK104GL",  UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK104GLM, "GK104GLM", UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK104M,   "GK104M",   UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK106,    "GK106",    UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK106GL,  "GK106GL",  UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK106GLM, "GK106GLM", UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK106M,   "GK106M",   UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK107,    "GK107",    UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK107GL,  "GK107GL",  UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK107GLM, "GK107GLM", UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK107M,   "GK107M",   UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK110,    "GK110",    UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK110B,   "GK110B",   UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK110BGL, "GK110BGL", UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK110GL,  "GK110GL",  UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK208,    "GK208",    UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK208B,   "GK208B",   UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK208BM,  "GK208BM",  UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK208GLM, "GK208GLM", UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK208M,   "GK208M",   UARCH_KEPLER,  28)
  CHECK_UARCH(arch, CHIP_GK210GL,  "GK210GL",  UARCH_KEPLER,  28)
  // MAXWELL (5.0, 5.2, 5.3)                                   //
  CHECK_UARCH(arch, CHIP_GM107,    "GM107",    UARCH_MAXWELL, 28)
  CHECK_UARCH(arch, CHIP_GM107GL,  "GM107GL",  UARCH_MAXWELL, 28)
  CHECK_UARCH(arch, CHIP_GM107GLM, "GM107GLM", UARCH_MAXWELL, 28)
  CHECK_UARCH(arch, CHIP_GM107M,   "GM107M",   UARCH_MAXWELL, 28)
  CHECK_UARCH(arch, CHIP_GM108GLM, "GM108GLM", UARCH_MAXWELL, 28)
  CHECK_UARCH(arch, CHIP_GM108M,   "GM108M",   UARCH_MAXWELL, 28)
  CHECK_UARCH(arch, CHIP_GM200,    "GM200",    UARCH_MAXWELL, 28)
  CHECK_UARCH(arch, CHIP_GM200GL,  "GM200GL",  UARCH_MAXWELL, 28)
  CHECK_UARCH(arch, CHIP_GM204,    "GM204",    UARCH_MAXWELL, 28)
  CHECK_UARCH(arch, CHIP_GM204GL,  "GM204GL",  UARCH_MAXWELL, 28)
  CHECK_UARCH(arch, CHIP_GM204GLM, "GM204GLM", UARCH_MAXWELL, 28)
  CHECK_UARCH(arch, CHIP_GM204M,   "GM204M",   UARCH_MAXWELL, 28)
  CHECK_UARCH(arch, CHIP_GM206,    "GM206",    UARCH_MAXWELL, 28)
  CHECK_UARCH(arch, CHIP_GM206GL,  "GM206GL",  UARCH_MAXWELL, 28)
  CHECK_UARCH(arch, CHIP_GM206GLM, "GM206GLM", UARCH_MAXWELL, 28)
  CHECK_UARCH(arch, CHIP_GM206M,   "GM206M",   UARCH_MAXWELL, 28)
  // PASCAL (6.0, 6.1, 6.2)                                    //
  CHECK_UARCH(arch, CHIP_GP100,    "GP100",    UARCH_PASCAL,  16)
  CHECK_UARCH(arch, CHIP_GP100GL,  "GP100GL",  UARCH_PASCAL,  16)
  CHECK_UARCH(arch, CHIP_GP102,    "GP102",    UARCH_PASCAL,  16)
  CHECK_UARCH(arch, CHIP_GP102GL,  "GP102GL",  UARCH_PASCAL,  16)
  CHECK_UARCH(arch, CHIP_GP104,    "GP104",    UARCH_PASCAL,  16)
  CHECK_UARCH(arch, CHIP_GP104BM,  "GP104BM",  UARCH_PASCAL,  16)
  CHECK_UARCH(arch, CHIP_GP104GL,  "GP104GL",  UARCH_PASCAL,  16)
  CHECK_UARCH(arch, CHIP_GP104GLM, "GP104GLM", UARCH_PASCAL,  16)
  CHECK_UARCH(arch, CHIP_GP104M,   "GP104M",   UARCH_PASCAL,  16)
  CHECK_UARCH(arch, CHIP_GP106,    "GP106",    UARCH_PASCAL,  16)
  CHECK_UARCH(arch, CHIP_GP106BM,  "GP106BM",  UARCH_PASCAL,  16)
  CHECK_UARCH(arch, CHIP_GP106GL,  "GP106GL",  UARCH_PASCAL,  16)
  CHECK_UARCH(arch, CHIP_GP106M,   "GP106M",   UARCH_PASCAL,  16)
  CHECK_UARCH(arch, CHIP_GP107,    "GP107",    UARCH_PASCAL,  14)
  CHECK_UARCH(arch, CHIP_GP107BM,  "GP107BM",  UARCH_PASCAL,  14)
  CHECK_UARCH(arch, CHIP_GP107GL,  "GP107GL",  UARCH_PASCAL,  14)
  CHECK_UARCH(arch, CHIP_GP107GLM, "GP107GLM", UARCH_PASCAL,  14)
  CHECK_UARCH(arch, CHIP_GP107M,   "GP107M",   UARCH_PASCAL,  14)
  CHECK_UARCH(arch, CHIP_GP108,    "GP108",    UARCH_PASCAL,  14)
  CHECK_UARCH(arch, CHIP_GP108BM,  "GP108BM",  UARCH_PASCAL,  14)
  CHECK_UARCH(arch, CHIP_GP108GLM, "GP108GLM", UARCH_PASCAL,  14)
  CHECK_UARCH(arch, CHIP_GP108M,   "GP108M",   UARCH_PASCAL,  14)
  // VOLTA (7.0, 7.2)                                          //
  CHECK_UARCH(arch, CHIP_GV100,    "GV100",    UARCH_VOLTA,   12)
  CHECK_UARCH(arch, CHIP_GV100GL,  "GV100GL",  UARCH_VOLTA,   12)
  // TURING (7.5)                                              //
  CHECK_UARCH(arch, CHIP_TU102,    "TU102",    UARCH_TURING,  12)
  CHECK_UARCH(arch, CHIP_TU102GL,  "TU102GL",  UARCH_TURING,  12)
  CHECK_UARCH(arch, CHIP_TU104,    "TU104",    UARCH_TURING,  12)
  CHECK_UARCH(arch, CHIP_TU104BM,  "TU104BM",  UARCH_TURING,  12)
  CHECK_UARCH(arch, CHIP_TU104GL,  "TU104GL",  UARCH_TURING,  12)
  CHECK_UARCH(arch, CHIP_TU104GLM, "TU104GLM", UARCH_TURING,  12)
  CHECK_UARCH(arch, CHIP_TU104M,   "TU104M",   UARCH_TURING,  12)
  CHECK_UARCH(arch, CHIP_TU106,    "TU106",    UARCH_TURING,  12)
  CHECK_UARCH(arch, CHIP_TU106BM,  "TU106BM",  UARCH_TURING,  12)
  CHECK_UARCH(arch, CHIP_TU106GLM, "TU106GLM", UARCH_TURING,  12)
  CHECK_UARCH(arch, CHIP_TU106M,   "TU106M",   UARCH_TURING,  12)
  CHECK_UARCH(arch, CHIP_TU116,    "TU116",    UARCH_TURING,  12)
  CHECK_UARCH(arch, CHIP_TU116BM,  "TU116BM",  UARCH_TURING,  12)
  CHECK_UARCH(arch, CHIP_TU116GL,  "TU116GL",  UARCH_TURING,  12)
  CHECK_UARCH(arch, CHIP_TU116M,   "TU116M",   UARCH_TURING,  12)
  CHECK_UARCH(arch, CHIP_TU117,    "TU117",    UARCH_TURING,  12)
  CHECK_UARCH(arch, CHIP_TU117BM,  "TU117BM",  UARCH_TURING,  12)
  CHECK_UARCH(arch, CHIP_TU117GL,  "TU117GL",  UARCH_TURING,  12)
  CHECK_UARCH(arch, CHIP_TU117GLM, "TU117GLM", UARCH_TURING,  12)
  CHECK_UARCH(arch, CHIP_TU117M,   "TU117M",   UARCH_TURING,  12)
  // AMPERE (8.0, 8.6)                                         //
  CHECK_UARCH(arch, CHIP_GA100,    "GA100",    UARCH_AMPERE,   7)
  CHECK_UARCH(arch, CHIP_GA100GL,  "GA100GL",  UARCH_AMPERE,   7)
  CHECK_UARCH(arch, CHIP_GA102,    "GA102",    UARCH_AMPERE,   8)
  CHECK_UARCH(arch, CHIP_GA102GL,  "GA102GL",  UARCH_AMPERE,   8)
  CHECK_UARCH(arch, CHIP_GA104,    "GA104",    UARCH_AMPERE,   8)
  CHECK_UARCH(arch, CHIP_GA104GL,  "GA104GL",  UARCH_AMPERE,   8)
  CHECK_UARCH(arch, CHIP_GA104GLM, "GA104GLM", UARCH_AMPERE,   8)
  CHECK_UARCH(arch, CHIP_GA104M,   "GA104M",   UARCH_AMPERE,   8)
  CHECK_UARCH(arch, CHIP_GA106,    "GA106",    UARCH_AMPERE,   8)
  CHECK_UARCH(arch, CHIP_GA106M,   "GA106M",   UARCH_AMPERE,   8)
  CHECK_UARCH(arch, CHIP_GA107,    "GA107",    UARCH_AMPERE,   8)
  CHECK_UARCH(arch, CHIP_GA107BM,  "GA107BM",  UARCH_AMPERE,   8)
  CHECK_UARCH(arch, CHIP_GA107GLM, "GA107GLM", UARCH_AMPERE,   8)
  CHECK_UARCH(arch, CHIP_GA107M,   "GA107M",   UARCH_AMPERE,   8)
  CHECK_UARCH_END
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
  arch->chip = get_chip_from_pci(gpu->pci);

  map_chip_to_uarch(arch);

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
  char* str = (char *) emalloc(sizeof(char) * (strlen(STRING_UNKNOWN)+1));
  int32_t process = arch->process;

  if(process == UNK) {
    snprintf(str, strlen(STRING_UNKNOWN)+1, STRING_UNKNOWN);
  }
  else if(process > 100) {
    sprintf(str, "%.2fum", (double)process/100);
  }
  else if(process > 0){
    sprintf(str, "%dnm", process);
  }
  else {
    snprintf(str, strlen(STRING_UNKNOWN)+1, STRING_UNKNOWN);
    printBug("Found invalid process: '%d'", process);
  }

  return str;
}

char* get_str_chip(struct uarch* arch) {
  return arch->chip_str;
}

void free_uarch_struct(struct uarch* arch) {

}
