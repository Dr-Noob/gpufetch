#ifndef __HSA_UARCH__
#define __HSA_UARCH__

#include "../common/gpu.hpp"
#include "../common/uarch.hpp"

struct uarch;

struct uarch* get_uarch_from_hsa(struct gpu_info* gpu, char* gpu_name);
char* get_str_uarch_hsa(struct uarch* arch);

#endif
