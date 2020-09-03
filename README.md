# GPU Performance Portability Samples

The samples have been tested on the following systems:

|        | Iris | Tulip  | local |
|--------|------|--------|-------|
| CUDA   |      |        | yes   |
| HIP    |      | yes    |       |
| Kokkos | \*   |        | yes   |
| OpenCL | yes  | broken |       |
| OpenMP | yes  | yes    |       |

\*: Currently in a branch, requires non-portable workarounds for Iris.
