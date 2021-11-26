#ifndef __COMMON_UARCH__
#define __COMMON_UARCH__

typedef uint32_t GPUCHIP;
typedef uint32_t MICROARCH;

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

#endif
