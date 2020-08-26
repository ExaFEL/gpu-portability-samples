#include <stdio.h>

__global__ void saxpy(const size_t num_elements, const float alpha,
                      const float *x, const float *y, float *z) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < num_elements) z[idx] += alpha * x[idx] + y[idx];
}

int main() {
  size_t num_elements = 1 << 20;
  size_t buffer_size = num_elements * sizeof(float);

  float *x = (float *)malloc(buffer_size);
  float *y = (float *)malloc(buffer_size);
  float *z = (float *)malloc(buffer_size);

  float *d_x, *d_y, *d_z;
  cudaMalloc(&d_x, buffer_size);
  cudaMalloc(&d_y, buffer_size);
  cudaMalloc(&d_z, buffer_size);

  for (size_t idx = 0; idx < num_elements; idx++) {
    x[idx] = 1.0f;
    y[idx] = 2.0f;
    z[idx] = 0.0f;
  }

  cudaMemcpy(d_x, x, buffer_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, buffer_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, z, buffer_size, cudaMemcpyHostToDevice);

  saxpy<<<(num_elements + 255) / 256, 256>>>(num_elements, 2.0f, d_x, d_y, d_z);

  cudaMemcpy(z, d_z, buffer_size, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  float error = 0.0;
  for (size_t idx = 0; idx < num_elements; idx++) {
    error = fmax(error, fabs(z[idx] - 4.0f));
  }
  printf("error: %e\n", error);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);

  free(x);
  free(y);
  free(z);

  return 0;
}
