#define NUM_BUCKETS 128

kernel void histogram(const ulong num_elements, float range,
		      global float *data, global uint *histogram) {
  uint t = get_local_id(0);
  uint nt = get_local_size(0);

  local unsigned local_histogram[NUM_BUCKETS];
  for (uint i = t; i < NUM_BUCKETS; i += nt) local_histogram[i] = 0;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint idx = get_global_id(0); idx < num_elements; idx += get_global_size(0)) {
    size_t bucket = floor(data[idx] / range * (NUM_BUCKETS - 1));
    atomic_add(&local_histogram[bucket], 1);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint i = t; i < NUM_BUCKETS; i += nt)
    atomic_add(&histogram[i], local_histogram[i]);
}
