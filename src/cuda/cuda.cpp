#include <helper_cuda.h>
#include <cuda_runtime.h>

#include "cuda.hpp"
#include "uarch.hpp"
#include "../common/pci.hpp"
#include "../common/global.hpp"

int print_gpus_list() {
  cudaError_t err = cudaSuccess;
  int num_gpus = -1;

  if ((err = cudaGetDeviceCount(&num_gpus)) != cudaSuccess) {
    printErr("%s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
    return EXIT_FAILURE;
  }
  printf("CUDA GPUs available: %d\n", num_gpus);

  if(num_gpus > 0) {
    cudaDeviceProp deviceProp;
    int max_len = 0;

    for(int idx=0; idx < num_gpus; idx++) {
      if ((err = cudaGetDeviceProperties(&deviceProp, idx)) != cudaSuccess) {
        printErr("%s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
        return EXIT_FAILURE;
      }
      max_len = max(max_len, (int) strlen(deviceProp.name));
    }

    for(int i=0; i < max_len + 32; i++) putchar('-');
    putchar('\n');
    for(int idx=0; idx < num_gpus; idx++) {
      if ((err = cudaGetDeviceProperties(&deviceProp, idx)) != cudaSuccess) {
        printErr("%s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
        return EXIT_FAILURE;
      }
      printf("GPU %d: %s (Compute Capability %d.%d)\n", idx, deviceProp.name, deviceProp.major, deviceProp.minor);
    }
  }

  return EXIT_SUCCESS;
}

struct cache* get_cache_info(cudaDeviceProp prop) {
  struct cache* cach = (struct cache*) emalloc(sizeof(struct cache));

  cach->L2 = (struct cach*) emalloc(sizeof(struct cach));
  cach->L2->size = prop.l2CacheSize;
  cach->L2->num_caches = 1;
  cach->L2->exists = true;

  return cach;
}

struct topology* get_topology_info(cudaDeviceProp prop) {
  struct topology* topo = (struct topology*) emalloc(sizeof(struct topology));

  topo->streaming_mp = prop.multiProcessorCount;
  topo->cores_per_mp = _ConvertSMVer2Cores(prop.major, prop.minor);
  topo->cuda_cores = topo->streaming_mp * topo->cores_per_mp;

  return topo;
}

int32_t guess_clock_multipilier(struct gpu_info* gpu, struct memory* mem) {
  // Guess clock multiplier
  int32_t clk_mul = 1;

  int32_t clk8 = abs((mem->freq/8) - gpu->freq);
  int32_t clk4 = abs((mem->freq/4) - gpu->freq);
  int32_t clk2 = abs((mem->freq/2) - gpu->freq);
  int32_t clk1 = abs((mem->freq/1) - gpu->freq);

  int32_t min = mem->freq;
  if(clkm_possible_for_uarch(8, gpu->arch) && min > clk8) { clk_mul = 8; min = clk8; }
  if(clkm_possible_for_uarch(4, gpu->arch) && min > clk4) { clk_mul = 4; min = clk4; }
  if(clkm_possible_for_uarch(2, gpu->arch) && min > clk2) { clk_mul = 2; min = clk2; }
  if(clkm_possible_for_uarch(1, gpu->arch) && min > clk1) { clk_mul = 1; min = clk1; }

  return clk_mul;
}

struct memory* get_memory_info(struct gpu_info* gpu, cudaDeviceProp prop) {
  struct memory* mem = (struct memory*) emalloc(sizeof(struct memory));

  mem->size_bytes = (unsigned long long) prop.totalGlobalMem;
  mem->freq = prop.memoryClockRate * 0.001f;
  mem->bus_width = prop.memoryBusWidth;
  mem->clk_mul = guess_clock_multipilier(gpu, mem);
  mem->type = guess_memtype_from_cmul_and_uarch(mem->clk_mul, gpu->arch);

  // Fix frequency returned from CUDA to show real frequency
  mem->freq = mem->freq  / mem->clk_mul;

  return mem;
}

int64_t get_peak_performance(struct gpu_info* gpu) {
  return gpu->freq * 1000000 * gpu->topo->cuda_cores * 2;
}

struct gpu_info* get_gpu_info(int gpu_idx) {
  struct gpu_info* gpu = (struct gpu_info*) emalloc(sizeof(struct gpu_info));
  gpu->pci = NULL;
  gpu->idx = gpu_idx;

  if(gpu->idx < 0) {
    printErr("GPU index must be equal or greater than zero");
    return NULL;
  }

  printf("Waiting for CUDA driver to start...");
  fflush(stdout);

  int num_gpus = -1;
  cudaError_t err = cudaSuccess;
  if ((err = cudaGetDeviceCount(&num_gpus)) != cudaSuccess) {
    printErr("%s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }
  printf("\r                                   ");

  if(num_gpus <= 0) {
    printErr("No CUDA capable devices found!");
    return NULL;
  }

  if(gpu->idx+1 > num_gpus) {
    printErr("Requested GPU index %d in a system with %d GPUs", gpu->idx, num_gpus);
    return NULL;
  }

  cudaDeviceProp deviceProp;
  if ((err = cudaGetDeviceProperties(&deviceProp, gpu->idx)) != cudaSuccess) {
    printErr("%s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  gpu->freq = deviceProp.clockRate * 1e-3f;
  gpu->vendor = GPU_VENDOR_NVIDIA;
  gpu->name = (char *) emalloc(sizeof(char) * (strlen(deviceProp.name) + 1));
  strcpy(gpu->name, deviceProp.name);

  struct pci_dev *devices = get_pci_devices_from_pciutils();
  gpu->pci = get_pci_from_pciutils(devices);
  gpu->arch = get_uarch_from_cuda(gpu);
  gpu->cach = get_cache_info(deviceProp);
  gpu->mem = get_memory_info(gpu, deviceProp);
  gpu->topo = get_topology_info(deviceProp);
  gpu->peak_performance = get_peak_performance(gpu);

  return gpu;
}

char* get_str_sm(struct gpu_info* gpu) {
  uint32_t max_size = 10;
  char* dummy = (char *) ecalloc(max_size, sizeof(char));
  snprintf(dummy, max_size, "%d", gpu->topo->streaming_mp);
  return dummy;
}

char* get_str_cores_sm(struct gpu_info* gpu) {
  uint32_t max_size = 10;
  char* dummy = (char *) ecalloc(max_size, sizeof(char));
  snprintf(dummy, max_size, "%d", gpu->topo->cores_per_mp);
  return dummy;
}

char* get_str_cuda_cores(struct gpu_info* gpu) {
  uint32_t max_size = 10;
  char* dummy = (char *) ecalloc(max_size, sizeof(char));
  snprintf(dummy, max_size, "%d", gpu->topo->cuda_cores);
  return dummy;
}

