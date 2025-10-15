#ifndef __INTEL_UARCH__
#define __INTEL_UARCH__

#include "../common/gpu.hpp"
#include "../common/uarch.hpp"

struct uarch;

struct uarch* get_uarch_from_pci(struct pci* pci);
char* get_name_from_uarch(struct uarch* arch);
char* get_str_gt(struct uarch* arch);
char* get_str_uarch_intel(struct uarch* arch);
struct topology_i* get_topology_info(struct uarch* arch);

#endif
