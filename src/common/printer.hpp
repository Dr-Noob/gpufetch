#ifndef __PRINTER__
#define __PRINTER__

typedef int STYLE;

#include "args.hpp"
#include "../cuda/cuda.hpp"

#define COLOR_DEFAULT_NVIDIA "15,125,194:230,230,230:40,150,220:230,230,230"

bool print_gpufetch(struct gpu_info* gpu, STYLE s, struct color** cs);

#endif
