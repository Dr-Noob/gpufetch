#ifndef __PCI__
#define __PCI__

#include <stdint.h>
#include "nvmlb.hpp"

struct pci;

struct pci* get_pci_from_nvml(struct nvml_data* data);

#endif
