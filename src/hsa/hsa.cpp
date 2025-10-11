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

struct gpu_info* get_gpu_info_hsa(struct pci_dev *devices, int gpu_idx) {
  struct gpu_info* gpu = (struct gpu_info*) emalloc(sizeof(struct gpu_info));
  gpu->pci = NULL;
  gpu->idx = gpu_idx;

  if(gpu->idx < 0) {
    printErr("GPU index must be equal or greater than zero");
    return NULL;
  }

  hsa_status_t status;

  // Initialize the HSA runtime
  status = hsa_init();
  if (status != HSA_STATUS_SUCCESS) {
    printErr("Failed to initialize HSA runtime");
    return NULL;
  }

  // Lambda for iterating over agents
  auto agent_callback = [](hsa_agent_t agent, void* data) -> hsa_status_t {
    hsa_device_type_t type;
    if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS)
      return HSA_STATUS_SUCCESS;

    if (type == HSA_DEVICE_TYPE_GPU) {
      char name[64] = {0};
      if (hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name) == HSA_STATUS_SUCCESS) {
        std::cout << name << std::endl;
      }
    }

    return HSA_STATUS_SUCCESS;
  };

  // Iterate over all agents in the system
  status = hsa_iterate_agents(agent_callback, nullptr);
  if (status != HSA_STATUS_SUCCESS) {
    std::cerr << "Failed to iterate HSA agents.\n";
    hsa_shut_down();
    return NULL;
  }

  // Shut down the HSA runtime
  hsa_shut_down();
  return NULL;
}
