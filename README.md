# GPU Performance Portability Samples

The samples have been tested on the following systems:

## Saxpy

|        | Iris | Tulip | Crusher | local |
|--------|------|-------|---------|-------|
| CUDA   |      |       |         | yes   |
| HIP    | yes  | yes   | yes     |       |
| Kokkos | yes  | yes   | yes     |       |
| OpenCL | yes  | yes   | yes     |       |
| OpenMP | yes  | yes   | fails   |       |

## Histogram

|        | Iris | Tulip | Crusher | local |
|--------|------|-------|---------|-------|
| CUDA   |      |       |         | yes   |
| HIP    | yes  | yes   | yes     |       |
| Kokkos | yes  | yes   | yes     |       |
| OpenCL | yes  | yes   | yes     |       |
| OpenMP | \*   | \*\*  | \*\*\*  |       |

\* Internal compiler error: `omp_pteam_mem_alloc` is not yet
supported. Can work around by commenting out the `allocator` line,
unclear if it's using shared memory in that case or not.

\*\* Internal compiler error.

\*\*\* `<unknown>:0: error: local_histogram: unsupported initializer for address space`
