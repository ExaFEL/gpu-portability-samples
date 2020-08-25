These instructions are intended for Iris.

To build:

```
cd gpu-portability-sample/openmp
source env.sh
make
```

To run:

```
qsub -I -n 1 -t 30 -q iris
source env.sh
./saxpy
```
