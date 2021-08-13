// NVML Backend
#ifndef __NVMLB__
#define __NVMLB__

#include <stdbool.h>
#include <stdint.h>

struct nvml_data;

struct nvml_data* nvml_init();
bool nvml_get_pci_info(int dev, struct nvml_data* data);
uint16_t nvml_get_pci_vendor_id(struct nvml_data* data);
uint16_t nvml_get_pci_device_id(struct nvml_data* data);
bool nvml_shutdown(struct nvml_data* data);

#endif
