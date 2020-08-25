#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

int main() {
  size_t num_elements = 1 << 20;
  size_t buffer_size = num_elements * sizeof(float);

  float *x = (float *)malloc(buffer_size);
  float *y = (float *)malloc(buffer_size);
  float *z = (float *)malloc(buffer_size);

  const float alpha = 2.0f;

  for (size_t idx = 0; idx < num_elements; idx++) {
    x[idx] = 1.0f;
    y[idx] = 2.0f;
    z[idx] = 0.0f;
  }

  // Note: if you have unified shared memory on the GPU, you can
  // replace the map clause with:
  //
  // #pragma omp requires unified_shared_memory

#pragma omp target map(x[:num_elements], y[:num_elements], z[:num_elements])
  {
#pragma omp teams distribute parallel for
    for (size_t idx = 0; idx < num_elements; idx++) {
      z[idx] += alpha * x[idx] + y[idx];
    }
  }

  float error = 0.0;
  for (size_t idx = 0; idx < num_elements; idx++) {
    error = fmax(error, fabs(z[idx] - 4.0f));
  }
  printf("error: %e\n", error);

  free(x);
  free(y);
  free(z);

  return 0;
}
