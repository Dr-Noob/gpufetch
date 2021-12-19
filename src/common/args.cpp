#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <climits>

#include "args.hpp"
#include "global.hpp"

#define OVERFLOW          -1
#define UNDERFLOW         -2
#define INVALID_ARG       -3
#define NUM_COLORS         4

#define COLOR_STR_NVIDIA "nvidia"
#define COLOR_STR_INTEL  "intel"

//                              +-----------------------+-----------------------+
//                              | Color logo            | Color text            |
//                              | Color 1   | Color 2   | Color 1   | Color 2   |
#define COLOR_DEFAULT_NVIDIA    "118,185,000:255,255,255:255,255,255:118,185,000"
#define COLOR_DEFAULT_INTEL     "015,125,194:230,230,230:040,150,220:230,230,230"

struct args_struct {
  bool help_flag;
  bool version_flag;
  bool list_gpus;
  int gpu_idx;
  STYLE style;
  struct color** colors;
};

int errn = 0;
static struct args_struct args;

const char args_chr[] = {
  /* [ARG_COLOR]   = */ 'c',
  /* [ARG_GPU]     = */ 'g',
  /* [ARG_LIST]    = */ 'l',
  /* [ARG_HELP]    = */ 'h',
  /* [ARG_VERSION] = */ 'V',
};

const char *args_str[] = {
  /* [ARG_COLOR]   = */ "color",
  /* [ARG_GPU]     = */ "gpu",
  /* [ARG_LIST]    = */ "list-gpus",
  /* [ARG_HELP]    = */ "help",
  /* [ARG_VERSION] = */ "version",
};

int getarg_int(char* str) {
  errn = 0;

  char* endptr;
  long tmp = strtol(str, &endptr, 10);

  if(*endptr) {
    errn = INVALID_ARG;
    return -1;
  }
  if(tmp == LONG_MIN) {
    errn = UNDERFLOW;
    return -1;
  }
  if(tmp == LONG_MAX) {
    errn = OVERFLOW;
    return -1;
  }
  if(tmp >= INT_MIN && tmp <= INT_MAX) {
    return (int)tmp;
  }

  errn = OVERFLOW;
  return -1;
}

void print_getarg_error() {
  switch (errn) {
    case OVERFLOW:
      printf("overflow detected while parsing the arguments\n");
      break;
    case UNDERFLOW:
      printf("underflow detected while parsing the arguments\n");
      break;
    case INVALID_ARG:
      printf("invalid argument\n");
      break;
    default:
      printf("invalid error: %d\n", errn);
      break;
  }
}

STYLE get_style() {
  return args.style;
}

struct color** get_colors() {
  return args.colors;
}

int get_gpu_idx() {
  return args.gpu_idx;
}

bool show_help() {
  return args.help_flag;
}

bool list_gpus() {
  return args.list_gpus;
}

bool show_version() {
  return args.version_flag;
}

int max_arg_str_length() {
  int max_len = -1;
  int len = sizeof(args_str) / sizeof(args_str[0]);
  for(int i=0; i < len; i++) {
    max_len = max(max_len, (int) strlen(args_str[i]));
  }
  return max_len;
}

char* build_short_options() {
  const char *c = args_chr;
  int len = sizeof(args_chr) / sizeof(args_chr[0]);
  char* str = (char *) emalloc(sizeof(char) * (len*2 + 1));
  memset(str, 0, sizeof(char) * (len*2 + 1));

  sprintf(str, "%c:%c:%c%c%c", c[ARG_GPU],
  c[ARG_COLOR], c[ARG_HELP], c[ARG_LIST],
  c[ARG_VERSION]);

  return str;
}

bool parse_color(char* optarg_str, struct color*** cs) {
  for(int i=0; i < NUM_COLORS; i++) {
    (*cs)[i] = (struct color *) emalloc(sizeof(struct color));
  }

  struct color** c = *cs;
  int32_t ret;
  char* str_to_parse = NULL;
  const char* color_to_copy = NULL;
  bool free_ptr = true;

  if(strcmp(optarg_str, COLOR_STR_NVIDIA) == 0) color_to_copy = COLOR_DEFAULT_NVIDIA;
  else if(strcmp(optarg_str, COLOR_STR_INTEL) == 0) color_to_copy = COLOR_DEFAULT_INTEL;
  else {
    str_to_parse = optarg_str;
    free_ptr = false;
  }

  if(str_to_parse == NULL) {
    str_to_parse = (char *) emalloc(sizeof(char) * (strlen(color_to_copy) + 1));
    strcpy(str_to_parse, color_to_copy);
  }

  ret = sscanf(str_to_parse, "%d,%d,%d:%d,%d,%d:%d,%d,%d:%d,%d,%d",
               &c[0]->R, &c[0]->G, &c[0]->B,
               &c[1]->R, &c[1]->G, &c[1]->B,
               &c[2]->R, &c[2]->G, &c[2]->B,
               &c[3]->R, &c[3]->G, &c[3]->B);

  if(ret != 12) {
    printErr("Expected to read 12 values for color but read %d", ret);
    return false;
  }

  for(int i=0; i < NUM_COLORS; i++) {
    if(c[i]->R < 0 || c[i]->R > 255) {
      printErr("Red in color %d is invalid: %d; must be in range (0, 255)", i+1, c[i]->R);
      return false;
    }
    if(c[i]->G < 0 || c[i]->G > 255) {
      printErr("Green in color %d is invalid: %d; must be in range (0, 255)", i+1, c[i]->G);
      return false;
    }
    if(c[i]->B < 0 || c[i]->B > 255) {
      printErr("Blue in color %d is invalid: %d; must be in range (0, 255)", i+1, c[i]->B);
      return false;
    }
  }

  if(free_ptr) free (str_to_parse);

  return true;
}

bool parse_args(int argc, char* argv[]) {
  int opt;
  int option_index = 0;
  opterr = 0;

  args.version_flag = false;
  args.help_flag = false;
  args.list_gpus = false;
  args.gpu_idx = 0;
  args.colors = NULL;

  const struct option long_options[] = {
    {args_str[ARG_COLOR],   required_argument, 0, args_chr[ARG_COLOR]   },
    {args_str[ARG_GPU],     required_argument, 0, args_chr[ARG_GPU]     },
    {args_str[ARG_LIST],    no_argument,       0, args_chr[ARG_LIST]    },
    {args_str[ARG_HELP],    no_argument,       0, args_chr[ARG_HELP]    },
    {args_str[ARG_VERSION], no_argument,       0, args_chr[ARG_VERSION] },
    {0, 0, 0, 0}
  };

  char* short_options = build_short_options();
  opt = getopt_long(argc, argv, short_options, long_options, &option_index);

  while (!args.help_flag && !args.version_flag && !args.list_gpus && opt != -1) {
    if(opt == args_chr[ARG_COLOR]) {
      args.colors = (struct color **) emalloc(sizeof(struct color *) * NUM_COLORS);
      if(!parse_color(optarg, &args.colors)) {
        return false;
      }
    }
    else if(opt == args_chr[ARG_GPU]) {
      args.gpu_idx = getarg_int(optarg);
      if(errn != 0) {
        printErr("Option %s: ", args_str[ARG_GPU]);
        print_getarg_error();
        args.help_flag  = true;
        return false;
      }
    }
    else if(opt == args_chr[ARG_LIST]) {
      args.list_gpus = true;
    }
    else if(opt == args_chr[ARG_HELP]) {
      args.help_flag = true;
    }
    else if(opt == args_chr[ARG_VERSION]) {
      args.version_flag = true;
    }
    else {
      printWarn("Invalid options");
      args.help_flag  = true;
    }

    option_index = 0;
    opt = getopt_long(argc, argv, short_options, long_options, &option_index);
  }

  if(optind < argc) {
    printWarn("Invalid options");
    args.help_flag  = true;
  }

  if((args.help_flag + args.version_flag) > 1) {
    printWarn("You should specify just one option");
    args.help_flag  = true;
  }

  return true;
}

void free_colors_struct(struct color** cs) {
  for(int i=0; i < NUM_COLORS; i++) {
    free(cs[i]);
  }
  free(cs);
}
