#include <stdint.h>
#include <cstddef>

typedef uint32_t MICROARCH;

// Data not available
#define NA                   -1

// Unknown manufacturing process
#define UNK                  -1

enum {
  UARCH_UNKNOWN,
  UARCH_TESLA
};

struct uarch {
  MICROARCH uarch;
  char* uarch_str;
  int32_t process;
};

struct uarch* get_uarch_from_cuda(struct gpu_info* gpu) {
  return NULL;
}

char* get_str_uarch(struct gpu_info* gpu) {
  return NULL;
}

char* get_str_process(struct gpu_info* gpu) {
  return NULL;
}

void free_uarch_struct(struct uarch* arch) {

}
