#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "args.hpp"
#include "global.hpp"
#include "master.hpp"
#include "../cuda/cuda.hpp"
#include "../cuda/uarch.hpp"

static const char* VERSION = "0.24";

void print_help(char *argv[]) {
  const char **t = args_str;
  const char *c = args_chr;
  int max_len = max_arg_str_length();

  printf("Usage: %s [OPTION]...\n", argv[0]);
  printf("Simple yet fancy GPU architecture fetching tool\n\n");

  printf("Options: \n");
  printf("  -%c, --%s %*s Set the color scheme (by default, gpufetch uses the system color scheme) See COLORS section for a more detailed explanation\n", c[ARG_COLOR], t[ARG_COLOR], (int) (max_len-strlen(t[ARG_COLOR])), "");
  printf("  -%c, --%s %*s List the available GPUs in the system\n", c[ARG_LIST], t[ARG_LIST], (int) (max_len-strlen(t[ARG_LIST])), "");
  printf("  -%c, --%s %*s Select the GPU to use (default: 0)\n", c[ARG_GPU], t[ARG_GPU], (int) (max_len-strlen(t[ARG_GPU])), "");
  printf("      --%s %*s Show the short version of the logo\n", t[ARG_LOGO_SHORT], (int) (max_len-strlen(t[ARG_LOGO_SHORT])), "");
  printf("      --%s %*s Show the long version of the logo\n", t[ARG_LOGO_LONG], (int) (max_len-strlen(t[ARG_LOGO_LONG])), "");
  printf("  -%c, --%s %*s Enable verbose output\n", c[ARG_VERBOSE], t[ARG_VERBOSE], (int) (max_len-strlen(t[ARG_VERBOSE])), "");
  printf("  -%c, --%s %*s Print this help and exit\n", c[ARG_HELP], t[ARG_HELP], (int) (max_len-strlen(t[ARG_HELP])), "");
  printf("  -%c, --%s %*s Print gpufetch version and exit\n", c[ARG_VERSION], t[ARG_VERSION], (int) (max_len-strlen(t[ARG_VERSION])), "");

  printf("\nCOLORS: \n");
  printf("  Color scheme can be set using a predefined color scheme or a custom one:\n");
  printf("  1. To use a predefined color scheme, the name of the scheme must be provided. Possible values are:\n");
  printf("  * \"nvidia\":  Use NVIDIA default color scheme \n");
  printf("  2. To use a custom color scheme, 4 colors must be given in RGB with the format: R,G,B:R,G,B:...\n");
  printf("  The first 2 colors are the GPU art color and the following 2 colors are the text colors\n");

  printf("\nEXAMPLES: \n");
  printf("  Run gpufetch with NVIDIA color scheme:\n");
  printf("    ./gpufetch --color nvidia\n");
  printf("  Run gpufetch with a custom color scheme:\n");
  printf("    ./gpufetch --color 239,90,45:210,200,200:100,200,45:0,200,200\n");

  printf("\nBUGS: \n");
  printf("    Report bugs to https://github.com/Dr-Noob/gpufetch/issues\n");

  printf("\nNOTE: \n");
  printf("    Peak performance information is NOT accurate. gpufetch computes peak performance using the max\n");
  printf("    frequency. However, to properly compute peak performance, you need to know the frequency of the\n");
  printf("    GPU running real code.\n");
  printf("    For peak performance measurement see: https://github.com/Dr-Noob/peakperf\n");
}

void print_version() {
  printf("gpufetch v%s\n", VERSION);
}

int main(int argc, char* argv[]) {
  if(!parse_args(argc,argv))
    return EXIT_FAILURE;

  if(show_help()) {
    print_help(argv);
    return EXIT_SUCCESS;
  }

  if(show_version()) {
    print_version();
    return EXIT_SUCCESS;
  }

  set_log_level(verbose_enabled());

  struct gpu_list* list = get_gpu_list();
  if(list_gpus()) {
    return print_gpus_list(list);
  }

  if(get_num_gpus_available(list) == 0) {
    printErr("No GPU was detected! Available GPUs are:");
    print_gpus_list_pci();
    printf("Please, make sure that the appropiate backend is enabled:\n");
    print_enabled_backends();
    printf("Visit https://github.com/Dr-Noob/gpufetch#2-backends for more information\n");

    return EXIT_FAILURE;
  }

  struct gpu_info* gpu = get_gpu_info(list, get_gpu_idx());
  if(gpu == NULL)
    return EXIT_FAILURE;

  printf("[NOTE]: gpufetch is in beta. The provided information may be incomplete or wrong.\n\
If you want to help to improve gpufetch, please compare the output of the program\n\
with a reliable source which you know is right (e.g, techpowerup.com) and report\n\
any inconsistencies to https://github.com/Dr-Noob/gpufetch/issues\n");

  if(print_gpufetch(gpu, get_style(), get_colors()))
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}
