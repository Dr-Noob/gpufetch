#ifndef __ARGS__
#define __ARGS__

#include <cstdint>

struct color {
  int32_t R;
  int32_t G;
  int32_t B;
};

enum {
  STYLE_EMPTY,
  STYLE_FANCY,
  STYLE_WILD,
  STYLE_RETRO,
  STYLE_LEGACY,
  STYLE_INVALID
};

enum {
  ARG_COLOR,
  ARG_GPU,
  ARG_LIST,
  ARG_LOGO_LONG,
  ARG_LOGO_SHORT,
  ARG_HELP,
  ARG_VERBOSE,
  ARG_VERSION
};

extern const char args_chr[];
extern const char *args_str[];

#include "printer.hpp"

int max_arg_str_length();
bool parse_args(int argc, char* argv[]);
bool show_help();
bool list_gpus();
bool show_logo_long();
bool show_logo_short();
bool show_version();
bool verbose_enabled();
void free_colors_struct(struct color** cs);
int get_gpu_idx();
struct color** get_colors();
STYLE get_style();

#endif
