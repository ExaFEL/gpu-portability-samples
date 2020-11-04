#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <omp.h>

#define NUM_BUCKETS 128

int main() {
  size_t num_elements = 1 << 20;
  size_t data_size = num_elements * sizeof(float);
  size_t histogram_size = num_elements * sizeof(NUM_BUCKETS);

  float *data = (float *)malloc(data_size);
  unsigned *histogram = (unsigned *)malloc(histogram_size);

  float range = (float)RAND_MAX;
  for (size_t idx = 0; idx < num_elements; idx++) {
    data[idx] = rand();
  }
  for (size_t idx = 0; idx < NUM_BUCKETS; idx++) {
    histogram[idx] = 0;
  }

#pragma omp target enter data map(alloc:                        \
                                  data[:num_elements],          \
                                  histogram[:NUM_BUCKETS])

#pragma omp target map(to: data[:num_elements]), map(tofrom: histogram[:NUM_BUCKETS])
  {
#pragma omp teams
    {
      unsigned local_histogram[NUM_BUCKETS];
#pragma omp allocate(local_histogram) allocator(omp_pteam_mem_alloc)

#pragma omp parallel
      {
        int t = omp_get_thread_num();
        int nt = omp_get_num_threads();

        for (int i = t; i < NUM_BUCKETS; i += nt) local_histogram[i] = 0;

#pragma omp barrier

        for (int idx = (omp_get_team_num() * omp_get_num_threads()) + omp_get_thread_num(); idx < num_elements;
             idx += omp_get_num_teams() * omp_get_num_threads()) {
          size_t bucket = floor(data[idx] / range * (NUM_BUCKETS - 1));
#pragma omp atomic
          local_histogram[bucket]++;
        }

#pragma omp barrier

        for (int i = t; i < NUM_BUCKETS; i += nt)
#pragma omp atomic
          histogram[i] += local_histogram[i];
      }
    }
  }

#pragma omp target exit data map(release:			\
                                 data[:num_elements],		\
                                 histogram[:num_elements])

  size_t total = 0;
  for (size_t idx = 0; idx < NUM_BUCKETS; idx++) {
    total += histogram[idx];
    printf("histogram[%lu] = %u\n", idx, histogram[idx]);
  }
  printf("\ntotal = %lu (%s)\n", total,
         total == num_elements ? "PASS" : "FAIL");

  free(data);
  free(histogram);

  return 0;
}
