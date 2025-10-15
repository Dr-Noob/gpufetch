#ifndef __HSA_GPUCHIPS__
#define __HSA_GPUCHIPS__

typedef uint32_t GPUCHIP;

enum {
  CHIP_UNKNOWN_HSA,
  // VEGA (TODO)
  // ...
  // RDNA
  CHIP_NAVI_10,
  CHIP_NAVI_12,
  CHIP_NAVI_14,
  // RDNA2
  // There are way more (eg Oberon)
  // Maybe we'll add them in the future.
  CHIP_NAVI_21,
  CHIP_NAVI_22,
  CHIP_NAVI_23,
  CHIP_NAVI_24,
  // RDNA3
  // There are way more as well.
  // Supporting Navi only for now.
  CHIP_NAVI_31,
  CHIP_NAVI_32,
  CHIP_NAVI_33,
  // RDNA4
  CHIP_NAVI_44,
  CHIP_NAVI_48,
  // CDNA
  CHIP_ARCTURUS,      // MI100 series
  CHIP_ALDEBARAN,     // MI200 series
  CHIP_AQUA_VANJARAM, // MI300 series
  CHIP_CDNA_NEXT      // MI350 series
};

#endif
