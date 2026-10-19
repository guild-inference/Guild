// Lifecycle fixture only: single-thread scan, not a tuned CUDA implementation.
#define SCAN_CUDA 1
#define SCAN_PACKAGE "example.scan.cuda"
#include "scan.c"
