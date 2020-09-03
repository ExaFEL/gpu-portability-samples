These instructions are intended for Iris or Tulip.

## Iris

To build:

```
cd gpu-portability-sample/opencl
source iris_env.sh
./build.sh
```

To run:

```
qsub -I -n 1 -t 30 -q iris
source iris_env.sh
./build/saxpy
```

## Tulip

To build:

```
cd gpu-portability-sample/opencl
source tulip_env.sh
./get_kokkos.sh
./build.sh
```

To run:

```
salloc -N 1 -p amdMI100 --time 00:30:00
srun ./build/saxpy
```
