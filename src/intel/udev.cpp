#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>

#include "../common/global.hpp"
#include "../common/pci.hpp"

#define _PATH_SYS_SYSTEM        "/sys/devices/pci0000:00"
#define _PATH_SYS_DRM           "/drm"
#define _PATH_CARD              "/card0"
#define _PATH_FREQUENCY_MAX     "/gt_max_freq_mhz"
#define _PATH_FREQUENCY_MIN     "/gt_min_freq_mhz"

#define _PATH_FREQUENCY_MAX_LEN 100
#define DEFAULT_FILE_SIZE       4096
#define UNKNOWN_DATA            -1

char* read_file(char* path, int* len) {
  int fd = open(path, O_RDONLY);

  if(fd == -1) {
    return NULL;
  }

  //File exists, read it
  int bytes_read = 0;
  int offset = 0;
  int block = 128;
  char* buf = (char *) emalloc(sizeof(char)*DEFAULT_FILE_SIZE);
  memset(buf, 0, sizeof(char)*DEFAULT_FILE_SIZE);

  while (  (bytes_read = read(fd, buf+offset, block)) > 0 ) {
    offset += bytes_read;
  }

  if (close(fd) == -1) {
    return NULL;
  }

  *len = offset;
  return buf;
}

long get_freq_from_file(char* path) {
  int filelen;
  char* buf;
  if((buf = read_file(path, &filelen)) == NULL) {
    printWarn("Could not open '%s'", path);
    return UNKNOWN_DATA;
  }

  char* end;
  errno = 0;
  long ret = strtol(buf, &end, 10);
  if(errno != 0) {
    printBug("strtol: %s", strerror(errno));
    free(buf);
    return UNKNOWN_DATA;
  }

  // We will be getting the frequency in MHz
  // We consider it is an error if frequency is
  // greater than 10 GHz or less than 100 MHz
  if(ret > 10000 || ret <  100) {
    printBug("Invalid data was read from file '%s': %ld\n", path, ret);
    return UNKNOWN_DATA;
  }

  free(buf);

  return ret;
}

long get_max_freq_from_file(struct pci* pci) {
  char path[_PATH_FREQUENCY_MAX_LEN];
  sprintf(path, "%s/%04x:%02x:%02x.%d%s%s%s", _PATH_SYS_SYSTEM, pci->domain, pci->bus, pci->dev, pci->func, _PATH_SYS_DRM, _PATH_CARD, _PATH_FREQUENCY_MAX);
  return get_freq_from_file(path);
}

long get_min_freq_from_file(struct pci* pci) {
  char path[_PATH_FREQUENCY_MAX_LEN];
  sprintf(path, "%s/%04x:%02x:%02x.%d%s%s%s", _PATH_SYS_SYSTEM, pci->domain, pci->bus, pci->dev, pci->func, _PATH_SYS_DRM, _PATH_CARD, _PATH_FREQUENCY_MIN);
  return get_freq_from_file(path);
}
