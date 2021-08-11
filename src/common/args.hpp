#ifndef __ARGS__
#define __ARGS__

#include <stdbool.h>

enum {
  ARG_HELP,
  ARG_VERSION
};

extern const char args_chr[];
extern const char *args_str[];

int max_arg_str_length();
bool parse_args(int argc, char* argv[]);
bool show_help();
bool show_version();

#endif
