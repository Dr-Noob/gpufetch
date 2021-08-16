#ifndef __PRINTER__
#define __PRINTER__

typedef int STYLE;

#include "args.hpp"
#include "../cuda/cuda.hpp"

bool print_gpufetch(struct gpu_info* gpu, STYLE s, struct color** cs);

#endif
