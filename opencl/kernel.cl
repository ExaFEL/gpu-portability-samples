kernel void saxpy(float alpha, global float *x, global float *y, global float *z) {
  uint idx = get_global_id(0);
  z[idx] += alpha * x[idx] + y[idx];
}
