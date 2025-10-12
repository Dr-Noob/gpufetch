#include <getopt.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <climits>

#include "args.hpp"
#include "global.hpp"

#define OVERFLOW          -1
#define UNDERFLOW         -2
#define INVALID_ARG       -3
#define NUM_COLORS         4

#define COLOR_STR_NVIDIA "nvidia"
#define COLOR_STR_AMD    "amd"
#define COLOR_STR_INTEL  "intel"

//                              +-----------------------+-----------------------+
//                              | Color logo            | Color text            |
//                              | Color 1   | Color 2   | Color 1   | Color 2   |
#define COLOR_DEFAULT_NVIDIA    "118,185,000:255,255,255:255,255,255:118,185,000"
#define COLOR_DEFAULT_AMD       "250,250,250:250,250,250:200,200,200:255,255,255"
#define COLOR_DEFAULT_INTEL     "015,125,194:230,230,230:040,150,220:230,230,230"

struct args_struct {
  bool help_flag;
  bool verbose_flag;
  bool version_flag;
  bool list_gpus;
  bool logo_long;
  bool logo_short;
  int gpu_idx;
  STYLE style;
  struct color** colors;
};

int errn = 0;
static struct args_struct args;

const char args_chr[] = {
  /* [ARG_COLOR]      = */ 'c',
  /* [ARG_GPU]        = */ 'g',
  /* [ARG_LIST]       = */ 'l',
  /* [ARG_LOGO_LONG]  = */ 1,
  /* [ARG_LOGO_SHORT] = */ 2,
  /* [ARG_HELP]       = */ 'h',
  /* [ARG_VERBOSE]    = */ 'v',
  /* [ARG_VERSION]    = */ 'V',
};

const char *args_str[] = {
  /* [ARG_COLOR]      = */ "color",
  /* [ARG_GPU]        = */ "gpu",
  /* [ARG_LIST]       = */ "list-gpus",
  /* [ARG_LOGO_LONG]  = */ "logo-long",
  /* [ARG_LOGO_SHORT] = */ "logo-short",
  /* [ARG_HELP]       = */ "help",
  /* [ARG_VERBOSE]    = */ "verbose",
  /* [ARG_VERSION]    = */ "version",
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

const char* getarg_error() {
  switch (errn) {
    case OVERFLOW:
      return "overflow detected";
    case UNDERFLOW:
      return "underflow detected";
    case INVALID_ARG:
      return "invalid argument";
    default:
      return "invalid error";
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

bool show_logo_long() {
  return args.logo_long;
}

bool show_logo_short() {
  return args.logo_short;
}

bool show_version() {
  return args.version_flag;
}

bool verbose_enabled() {
  return args.verbose_flag;
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

  sprintf(str, "%c:%c:%c%c%c%c%c%c", c[ARG_GPU],
  c[ARG_COLOR], c[ARG_HELP], c[ARG_LIST],
  c[ARG_LOGO_SHORT], c[ARG_LOGO_LONG],
  c[ARG_VERBOSE], c[ARG_VERSION]);

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
  else if(strcmp(optarg_str, COLOR_STR_AMD) == 0) color_to_copy = COLOR_DEFAULT_AMD;
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
  args.logo_long = false;
  args.logo_short = false;
  args.gpu_idx = 0;
  args.colors = NULL;

  const struct option long_options[] = {
    {args_str[ARG_COLOR],      required_argument, 0, args_chr[ARG_COLOR]      },
    {args_str[ARG_GPU],        required_argument, 0, args_chr[ARG_GPU]        },
    {args_str[ARG_LIST],       no_argument,       0, args_chr[ARG_LIST]       },
    {args_str[ARG_LOGO_SHORT], no_argument,       0, args_chr[ARG_LOGO_SHORT] },
    {args_str[ARG_LOGO_LONG],  no_argument,       0, args_chr[ARG_LOGO_LONG]  },
    {args_str[ARG_HELP],       no_argument,       0, args_chr[ARG_HELP]       },
    {args_str[ARG_VERBOSE],    no_argument,       0, args_chr[ARG_VERBOSE]    },
    {args_str[ARG_VERSION],    no_argument,       0, args_chr[ARG_VERSION]    },
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
      // Check for "a" option
      if(strcmp(optarg, "a") == 0) {
        args.gpu_idx = -1;
      }
      else {
        args.gpu_idx = getarg_int(optarg);
        if(errn != 0) {
          printErr("Option %s: %s", args_str[ARG_GPU], getarg_error());
          args.help_flag  = true;
          return false;
        }
        if(args.gpu_idx < 0) {
          printErr("Specified GPU index is out of range: %d. ", args.gpu_idx);
          printf("Run gpufetch with the --%s option to check out valid GPU indexes\n", args_str[ARG_LIST]);
          return false;
        }
      }
    }
    else if(opt == args_chr[ARG_LIST]) {
      args.list_gpus = true;
    }
    else if(opt == args_chr[ARG_LOGO_SHORT]) {
       args.logo_short = true;
    }
    else if(opt == args_chr[ARG_LOGO_LONG]) {
       args.logo_long = true;
    }
    else if(opt == args_chr[ARG_HELP]) {
      args.help_flag = true;
    }
    else if(opt == args_chr[ARG_VERBOSE]) {
      args.verbose_flag  = true;
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

  if(args.logo_short && args.logo_long) {
    printWarn("%s and %s cannot be specified together", args_str[ARG_LOGO_SHORT], args_str[ARG_LOGO_LONG]);
    args.logo_short = false;
    args.logo_long = false;
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
