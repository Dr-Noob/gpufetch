#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "args.h"
#include "global.h"

struct args_struct {
  bool help_flag;
  bool version_flag;
};

const char args_chr[] = {
  /* [ARG_CHAR_HELP]    = */ 'h',
  /* [ARG_CHAR_VERSION] = */ 'V',
};

const char *args_str[] = {
  /* [ARG_CHAR_HELP]    = */ "help",
  /* [ARG_CHAR_VERSION] = */ "version",
};

static struct args_struct args;

bool show_help() {
  return args.help_flag;
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

  sprintf(str, "%c%c",
  c[ARG_HELP], c[ARG_VERSION]);

  return str;
}

bool parse_args(int argc, char* argv[]) {
  int opt;
  int option_index = 0;
  opterr = 0;

  args.version_flag = false;
  args.help_flag = false;

  const struct option long_options[] = {
    {args_str[ARG_HELP],    no_argument,       0, args_chr[ARG_HELP]    },
    {args_str[ARG_VERSION], no_argument,       0, args_chr[ARG_VERSION] },
    {0, 0, 0, 0}
  };

  char* short_options = build_short_options();
  opt = getopt_long(argc, argv, short_options, long_options, &option_index);

  while (!args.help_flag && !args.version_flag && opt != -1) {
    if(opt == args_chr[ARG_HELP]) {
      args.help_flag  = true;
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
