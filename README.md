# GPU Performance Portability Samples

The samples have been tested on the following systems:

## Saxpy

|        | Iris | Tulip  | local |
|--------|------|--------|-------|
| CUDA   |      |        | yes   |
| HIP    |      | yes    |       |
| Kokkos | yes  | yes    |       |
| OpenCL | yes  | \*\*   |       |
| OpenMP | yes  | yes    |       |

\*\* Broken on Tulip, currently investigating.
