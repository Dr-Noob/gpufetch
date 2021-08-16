#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "args.hpp"
#include "global.hpp"
#include "../cuda/cuda.hpp"
#include "../cuda/uarch.hpp"

static const char* VERSION = "0.04";

void print_help(char *argv[]) {
  const char **t = args_str;
  const char *c = args_chr;
  int max_len = max_arg_str_length();

  printf("Usage: %s [OPTION]...\n", argv[0]);
  printf("Simple yet fancy GPU architecture fetching tool\n\n");

  printf("Options: \n");
  printf("  -%c, --%s %*s Prints this help and exit\n", c[ARG_HELP], t[ARG_HELP], (int) (max_len-strlen(t[ARG_HELP])), "");
  printf("  -%c, --%s %*s Prints gpufetch version and exit\n", c[ARG_VERSION], t[ARG_VERSION], (int) (max_len-strlen(t[ARG_VERSION])), "");

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

  set_log_level(true);

  printWarn("gpufetch is in beta. The provided information may be incomplete or wrong.\n\
           If you want to help to improve gpufetch, please compare the output of the program\n\
           with a reliable source which you know is right (e.g, techpowerup.com) and report\n\
           any inconsistencies to https://github.com/Dr-Noob/gpufetch/issues");

  struct gpu_info* gpu = get_gpu_info();
  if(gpu == NULL)
    return EXIT_FAILURE;

  if(print_gpufetch(gpu, get_style(), get_colors()))
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}
