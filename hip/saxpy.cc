#include <stdio.h>

#include <hip/hip_runtime.h>

__global__ void saxpy(const size_t num_elements, const float alpha,
                      const float *x, const float *y, float *z) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx <= num_elements) z[idx] += alpha * x[idx] + y[idx];
}

int main() {
  size_t num_elements = 1 << 20;
  size_t buffer_size = num_elements * sizeof(float);

  float *x = (float *)malloc(buffer_size);
  float *y = (float *)malloc(buffer_size);
  float *z = (float *)malloc(buffer_size);

  float *d_x, *d_y, *d_z;
  hipMalloc(&d_x, buffer_size);
  hipMalloc(&d_y, buffer_size);
  hipMalloc(&d_z, buffer_size);

  for (size_t idx = 0; idx < num_elements; idx++) {
    x[idx] = 1.0f;
    y[idx] = 2.0f;
    z[idx] = 0.0f;
  }

  hipMemcpy(d_x, x, buffer_size, hipMemcpyHostToDevice);
  hipMemcpy(d_y, y, buffer_size, hipMemcpyHostToDevice);
  hipMemcpy(d_z, z, buffer_size, hipMemcpyHostToDevice);

  saxpy<<<(num_elements + 255) / 256, 256>>>(num_elements, 2.0f, d_x, d_y, d_z);

  hipMemcpy(z, d_z, buffer_size, hipMemcpyDeviceToHost);

  hipDeviceSynchronize();

  float error = 0.0;
  for (size_t idx = 0; idx < num_elements; idx++) {
    error = fmax(error, fabs(z[idx] - 4.0f));
  }
  printf("error: %e\n", error);

  hipFree(d_x);
  hipFree(d_y);
  hipFree(d_z);

  free(x);
  free(y);
  free(z);

  return 0;
}
