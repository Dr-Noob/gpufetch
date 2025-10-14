#include <iostream>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstring>
#include <cstdlib>
#include <cstdio>

#include <iostream>
#include <iomanip>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include "hsa.hpp"
#include "uarch.hpp"
#include "../common/pci.hpp"
#include "../common/global.hpp"
#include "../common/uarch.hpp"

struct agent_info {
  unsigned deviceId; // ID of the target GPU device
  char gpu_name[64];  
  char vendor_name[64];
  char device_mkt_name[64];
  uint32_t max_clock_freq;
  uint32_t compute_unit;
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
    printErr("HSA failure at: %s:%d\n", __FILE__, __LINE__);                  \
    printErr("Call returned %s\n", err_str);                                  \
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

    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_VENDOR_NAME, info->vendor_name);
    RET_IF_HSA_ERR(err);

    err = hsa_agent_get_info(agent, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_PRODUCT_NAME, &info->device_mkt_name);
    RET_IF_HSA_ERR(err);

    err = hsa_agent_get_info(agent, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY, &info->max_clock_freq);
    RET_IF_HSA_ERR(err);

    err = hsa_agent_get_info(agent, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &info->compute_unit);
    RET_IF_HSA_ERR(err);
  }

  return HSA_STATUS_SUCCESS;
}

struct topology_h* get_topology_info(struct agent_info info) {
  struct topology_h* topo = (struct topology_h*) emalloc(sizeof(struct topology_h));

  topo->compute_units = info.compute_unit;

  return topo;
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

  hsa_status_t err = hsa_init();
  if (err != HSA_STATUS_SUCCESS) {
    printErr("Failed to initialize HSA runtime");
    return NULL;
  }

  struct agent_info info;
  info.deviceId = gpu_idx;

  // Iterate over all agents in the system
  err = hsa_iterate_agents(agent_callback, &info);
  if (err != HSA_STATUS_SUCCESS) {
    printErr("Failed to iterate HSA agents");
    hsa_shut_down();
    return NULL;
  }

  if (info.vendor_name != "AMD") {
    printErr("HSA vendor name is: '%s'. Only AMD is supported!", info.vendor_name);
    return NULL;
  }
  gpu->vendor = GPU_VENDOR_AMD;

  gpu->freq = info.max_clock_freq;
  gpu->topo_h = get_topology_info(info);
  gpu->name = (char *) emalloc(sizeof(char) * (strlen(info.device_mkt_name) + 1));
  strcpy(gpu->name, info.device_mkt_name);
  gpu->arch = get_uarch_from_hsa(gpu);

  if (gpu->arch->TARGET_UNKNOWN_HSA) {
    printErr("Unknown LLVM target: '%s'", gpu->name);
    return NULL;
  }

  // Shut down the HSA runtime
  err = hsa_shut_down();
  if (err != HSA_STATUS_SUCCESS) {
    printErr("Failed to shutdown HSA runtime");
    return NULL;
  }
  return gpu;
}

char* get_str_cu(struct gpu_info* gpu) {
  return get_str_generic(gpu->topo_h->compute_units);
}
