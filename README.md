# GPU Performance Portability Samples

The samples have been tested on the following systems:

## Saxpy

|        | Iris | Tulip  | local |
|--------|------|--------|-------|
| CUDA   |      |        | yes   |
| HIP    | yes† | yes    |       |
| Kokkos | yes  | yes    |       |
| OpenCL | yes  | yes    |       |
| OpenMP | yes  | yes    |       |

## Histogram

|        | Iris | Tulip  | local |
|--------|------|--------|-------|
| CUDA   |      |        | yes   |
| HIP    | ††   | yes    |       |
| Kokkos | \*   | yes    |       |
| OpenCL | yes  | yes    |       |
| OpenMP | \*\* | \*\*\* |       |

\* Builds, but fails test.

\*\* Internal compiler error: `omp_pteam_mem_alloc` is not yet
supported. Can work around by commenting out the `allocator` line,
unclear if it's using shared memory in that case or not.

\*\*\* Internal compiler error.

† Note: Requires workaround for `math.h`.

†† Builds, but fails test.
