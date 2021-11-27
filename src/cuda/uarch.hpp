#ifndef __CUDA_UARCH__
#define __CUDA_UARCH__

#include "../common/gpu.hpp"

struct uarch;

struct uarch* get_uarch_from_cuda(struct gpu_info* gpu);
bool clkm_possible_for_uarch(int clkm, struct uarch* arch);
MEMTYPE guess_memtype_from_cmul_and_uarch(int ddr, struct uarch* arch);
char* get_str_uarch_cuda(struct uarch* arch);
char* get_str_cc(struct uarch* arch);
char* get_str_chip(struct uarch* arch);
char* get_str_process(struct uarch* arch);
void free_uarch_struct(struct uarch* arch);

#endif
