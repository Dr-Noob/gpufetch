#include <nvml.h>

#include "nvmlb.hpp"
#include "../common/global.hpp"

struct nvml_data {
  bool nvml_started;
  nvmlPciInfo_t pci;
};

struct nvml_data* nvml_init() {
  struct nvml_data* data = (struct nvml_data*) emalloc(sizeof(struct nvml_data));
  data->nvml_started = false;

  nvmlReturn_t result;

  if ((result = nvmlInit()) != NVML_SUCCESS) {
    printErr("nvmlInit: %s\n", nvmlErrorString(result));
    return NULL;
  }

  data->nvml_started = true;
  return data;
}

bool nvml_get_pci_info(int dev, struct nvml_data* data) {
  nvmlReturn_t result;
  nvmlDevice_t device;

  if(!data->nvml_started) {
    printErr("nvml_get_pci_info: nvml was not started");
    return false;
  }

  if ((result = nvmlDeviceGetHandleByIndex(dev, &device)) != NVML_SUCCESS) {
    printErr("nvmlDeviceGetHandleByIndex: %s\n", nvmlErrorString(result));
    return false;
  }

  if ((result = nvmlDeviceGetPciInfo(device, &data->pci)) != NVML_SUCCESS) {
    printErr("nvmlDeviceGetPciInfo: %s\n", nvmlErrorString(result));
    return false;
  }

  return true;
}

uint16_t nvml_get_pci_vendor_id(struct nvml_data* data) {
  return data->pci.pciDeviceId & 0x0000FFFF;
}

uint16_t nvml_get_pci_device_id(struct nvml_data* data) {
  return (data->pci.pciDeviceId & 0xFFFF0000) >> 16;
}

bool nvml_shutdown(struct nvml_data* data) {
  nvmlReturn_t result;

  if(!data->nvml_started) {
    printWarn("nvml_get_pci_info: nvml was not started");
    return true;
  }

  if ((result = nvmlShutdown()) != NVML_SUCCESS) {
    printErr("nvmlShutdown: %s\n", nvmlErrorString(result));
    return false;
  }

  return true;
}
