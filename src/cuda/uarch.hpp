#ifndef __UARCH__
#define __UARCH__

#include "../common/gpu.hpp"

struct uarch;

struct uarch* get_uarch_from_cuda(struct gpu_info* gpu);
MEMTYPE guess_memtype_from_cmul_and_uarch(int ddr, struct uarch* arch);
char* get_str_uarch(struct uarch* arch);
char* get_str_cc(struct uarch* arch);
char* get_str_chip(struct uarch* arch);
char* get_str_process(struct uarch* arch);
void free_uarch_struct(struct uarch* arch);

#endif
