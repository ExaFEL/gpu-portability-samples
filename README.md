# GPU Performance Portability Samples

The samples have been tested on the following systems:

## Saxpy

|        | Iris | Tulip  | local |
|--------|------|--------|-------|
| CUDA   |      |        | yes   |
| HIP    |      | yes    |       |
| Kokkos | \*   |        | yes   |
| OpenCL | yes  | \*\*   |       |
| OpenMP | yes  | yes    |       |

\* Currently in a branch, requires non-portable workarounds for Iris.

\*\* Broken on Tulip, currently investigating.
