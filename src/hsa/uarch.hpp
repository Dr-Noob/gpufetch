#ifndef __HSA_UARCH__
#define __HSA_UARCH__

#include "../common/gpu.hpp"

struct uarch;

struct uarch* get_uarch_from_hsa(struct gpu_info* gpu, char* gpu_name);
char* get_str_uarch_hsa(struct uarch* arch);
char* get_str_process(struct uarch* arch); // TODO: Shouldnt we define this in the cpp?
char* get_str_chip(struct uarch* arch);

#endif
