#include <cstdio>
#include <cstdlib>
#include "pci.hpp"
#include "global.hpp"

// Code inspired in lspci.c
int compare_them(const void *A, const void *B) {
  const struct pci_dev *a = *(struct pci_dev **) A;
  const struct pci_dev *b = *(struct pci_dev **) B;

  if (a->domain < b->domain)
    return -1;
  if (a->domain > b->domain)
    return 1;
  if (a->bus < b->bus)
    return -1;
  if (a->bus > b->bus)
    return 1;
  if (a->dev < b->dev)
    return -1;
  if (a->dev > b->dev)
    return 1;
  if (a->func < b->func)
    return -1;
  if (a->func > b->func)
    return 1;

  return 0;
}

void sort_pci_devices(struct pci_dev **devices) {
  int i = 0;
  struct pci_dev **arr;

  int cnt = 0;
  for(struct pci_dev *dev=*devices; dev != NULL; dev=dev->next) {
    cnt++;
  }

  arr = (struct pci_dev **) emalloc(sizeof(struct pci_dev *) * cnt);
  for(struct pci_dev *dev=*devices; dev != NULL; dev=dev->next) {
    arr[i] = dev;
    i++;
  }

  qsort(arr, cnt, sizeof(struct pci_dev *), compare_them);

  struct pci_dev *ptr = *devices;
  struct pci_dev *ptrb = *devices;
  for(i = 0; i < cnt; i++) {
    ptr = arr[i];
    if(i > 0) {
      ptrb->next = ptr;
    }
    ptrb = ptr;
  }

  ptr->next = NULL;
  *devices = arr[0];
  free(arr);
}
