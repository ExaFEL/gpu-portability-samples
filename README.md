# GPU Performance Portability Samples

The samples have been tested on the following systems:

## Saxpy

|        | Iris | Tulip  | local |
|--------|------|--------|-------|
| CUDA   |      |        | yes   |
| HIP    |      | yes    |       |
| Kokkos | yes  | yes    |       |
| OpenCL | yes  | yes    |       |
| OpenMP | yes  | yes    |       |

## Histogram

|        | Iris | Tulip  | local |
|--------|------|--------|-------|
| CUDA   |      |        | yes   |
| HIP    |      | yes    |       |
| Kokkos | \*   | yes    |       |
| OpenCL | yes  | yes    |       |
| OpenMP |      |        |       |

\* Hits a compiler error, waiting for updated software release from Intel.
