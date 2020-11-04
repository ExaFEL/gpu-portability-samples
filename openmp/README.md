These instructions are intended for Iris or Tulip.

## Iris

To build:

```
cd gpu-portability-sample/openmp
source iris_env.sh
make
```

To run:

```
qsub -I -n 1 -t 30 -q iris
source iris_env.sh
./saxpy
```

## Tulip

To build:

```
cd gpu-portability-sample/openmp
source tulip_env.sh
make
```

To run:

```
salloc -N 1 -p amdMI100 --time 00:30:00
srun ./saxpy
```
