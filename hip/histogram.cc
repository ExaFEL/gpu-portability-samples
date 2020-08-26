#include <stdio.h>

#include <hip/hip_runtime.h>

#define NUM_BUCKETS 128

__global__ void compute_histogram(const size_t num_elements, const float range,
                                  const float *data, unsigned *histogram) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  int t = threadIdx.x;
  int nt = blockDim.x;

  __shared__ unsigned local_histogram[NUM_BUCKETS];
  for (int i = t; i < NUM_BUCKETS; i += nt) local_histogram[i] = 0;

  __syncthreads();

  if (idx <= num_elements) {
    size_t bucket = floor(data[idx] / range * (NUM_BUCKETS - 1));
    atomicAdd(&local_histogram[bucket], 1);
  }

  __syncthreads();

  for (int i = t; i < NUM_BUCKETS; i += nt)
    atomicAdd(&histogram[i], local_histogram[i]);
}

int main() {
  size_t num_elements = 1 << 20;
  size_t data_size = num_elements * sizeof(float);
  size_t histogram_size = NUM_BUCKETS * sizeof(unsigned);

  float *data = (float *)malloc(data_size);
  unsigned *histogram = (unsigned *)malloc(histogram_size);

  float *d_data;
  unsigned *d_histogram;
  hipMalloc(&d_data, data_size);
  hipMalloc(&d_histogram, histogram_size);

  float range = (float)RAND_MAX;
  for (size_t idx = 0; idx < num_elements; idx++) {
    data[idx] = rand();
  }
  for (size_t idx = 0; idx < NUM_BUCKETS; idx++) {
    histogram[idx] = 0;
  }

  hipMemcpy(d_data, data, data_size, hipMemcpyHostToDevice);
  hipMemcpy(d_histogram, histogram, histogram_size, hipMemcpyHostToDevice);

  compute_histogram<<<(num_elements + 255) / 256, 256>>>(num_elements, range,
                                                         d_data, d_histogram);

  hipMemcpy(histogram, d_histogram, histogram_size, hipMemcpyDeviceToHost);

  hipDeviceSynchronize();

  for (size_t idx = 0; idx < NUM_BUCKETS; idx++) {
    printf("histogram[%lu] = %u\n", idx, histogram[idx]);
  }

  hipFree(d_data);
  hipFree(d_histogram);

  free(data);
  free(histogram);

  return 0;
}
