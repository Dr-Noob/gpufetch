#include <iostream>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstring>
#include <cstdlib>
#include <cstdio>

#include "hsa.hpp"
#include "../common/pci.hpp"
#include "../common/global.hpp"
#include "../common/uarch.hpp"

struct agent_info {
  unsigned deviceId; // ID of the target GPU device
  char gpu_name[64];
};

#define RET_IF_HSA_ERR(err) { \
  if ((err) != HSA_STATUS_SUCCESS) { \
    char err_val[12];                                                         \
    char* err_str = NULL;                                                     \
    if (hsa_status_string(err,                                                \
            (const char**)&err_str) != HSA_STATUS_SUCCESS) {                  \
      snprintf(&(err_val[0]), sizeof(err_val), "%#x", (uint32_t)err);         \
      err_str = &(err_val[0]);                                                \
    }                                                                         \
    printErr("HSA failure at: %s:%d\n",                              \
                      __FILE__, __LINE__);                           \
    printErr("Call returned %s\n", err_str);                         \
    return (err);                                                             \
  }                                                                           \
}

hsa_status_t agent_callback(hsa_agent_t agent, void *data) {
  struct agent_info* info = reinterpret_cast<struct agent_info *>(data);

  hsa_device_type_t type;
  hsa_status_t err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  RET_IF_HSA_ERR(err);

  if (type == HSA_DEVICE_TYPE_GPU) {
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, info->gpu_name);
    RET_IF_HSA_ERR(err);
  }

  return HSA_STATUS_SUCCESS;
}

struct gpu_info* get_gpu_info_hsa(struct pci_dev *devices, int gpu_idx) {
  struct gpu_info* gpu = (struct gpu_info*) emalloc(sizeof(struct gpu_info));
  gpu->pci = NULL;
  gpu->idx = gpu_idx;

  if(gpu->idx < 0) {
    printErr("GPU index must be equal or greater than zero");
    return NULL;
  }

  if(gpu->idx > 0) {
    // Currently we only support fetching GPU 0.
    return NULL;
  }

  hsa_status_t status;

  // Initialize the HSA runtime
  status = hsa_init();
  if (status != HSA_STATUS_SUCCESS) {
    printErr("Failed to initialize HSA runtime");
    return NULL;
  }

  struct agent_info info;
  info.deviceId = gpu_idx;

  // Iterate over all agents in the system
  status = hsa_iterate_agents(agent_callback, &info);
  if (status != HSA_STATUS_SUCCESS) {
    printErr("Failed to iterate HSA agents");
    hsa_shut_down();
    return NULL;
  }

  gpu->vendor = GPU_VENDOR_AMD;
  gpu->name = (char *) emalloc(sizeof(char) * (strlen(info.gpu_name) + 1));
  strcpy(gpu->name, info.gpu_name);

  // Shut down the HSA runtime
  hsa_shut_down();
  return gpu;
}
