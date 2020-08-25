#include <stdio.h>

__global__
void saxpy(const size_t num_elements,
           const float alpha,
           const float *x, const float *y, float *z)
{
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx <= num_elements)
    z[idx] += alpha * x[idx] + y[idx];
}

int main()
{
  size_t num_elements = 1 << 20;
  float *x, *y, *z;
  float *d_x, *d_y, *d_z;

  x = (float *)malloc(num_elements * sizeof(float));
  y = (float *)malloc(num_elements * sizeof(float));
  z = (float *)malloc(num_elements * sizeof(float));

  cudaMalloc(&d_x, num_elements * sizeof(float));
  cudaMalloc(&d_y, num_elements * sizeof(float));
  cudaMalloc(&d_z, num_elements * sizeof(float));

  for (size_t idx = 0; idx < num_elements; idx++) {
    x[idx] = 1.0f;
    y[idx] = 2.0f;
    z[idx] = 0.0f;
  }

  cudaMemcpy(d_x, x, num_elements * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, num_elements * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, z, num_elements * sizeof(float), cudaMemcpyHostToDevice);

  saxpy<<<(num_elements+255)/256, 256>>>(num_elements, 2.0f, d_x, d_y, d_z);

  cudaMemcpy(z, d_z, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  float error = 0.0;
  for (size_t idx = 0; idx < num_elements; idx++) {
    error = max(error, abs(z[idx] - 4.0f));
  }
  printf("error: %e\n", error);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);

  free(x);
  free(y);
  free(z);
}
