#include <cstdint>
#include <cstdio>
#include <cstring>

#include "global.hpp"
#include "uarch.hpp"

char* get_str_process(struct uarch* arch) {
  char* str = (char *) emalloc(sizeof(char) * (strlen(STRING_UNKNOWN)+1));
  int32_t process = arch->process;

  if(process == UNK) {
    snprintf(str, strlen(STRING_UNKNOWN)+1, STRING_UNKNOWN);
  }
  else if(process > 100) {
    sprintf(str, "%.2fum", (double)process/100);
  }
  else if(process > 0){
    sprintf(str, "%dnm", process);
  }
  else {
    snprintf(str, strlen(STRING_UNKNOWN)+1, STRING_UNKNOWN);
    printBug("Found invalid process: '%d'", process);
  }

  return str;
}

