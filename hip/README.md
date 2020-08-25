These instructions are intended for Tulip.

To build:

```
cd gpu-portability-sample/hip
source env.sh
make
```

To run:

```
salloc -N 1 -p amdMI100 --time 00:10:00
srun ./saxpy
```
