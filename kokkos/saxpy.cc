#include <Kokkos_Core.hpp>

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);

  size_t num_elements = 1 << 20;
  size_t buffer_size = num_elements * sizeof(float);

  {
    Kokkos::View<float *> d_x("x", num_elements);
    Kokkos::View<float *> d_y("y", num_elements);
    Kokkos::View<float *> d_z("z", num_elements);

    Kokkos::View<float *>::HostMirror x = Kokkos::create_mirror_view(d_x);
    Kokkos::View<float *>::HostMirror y = Kokkos::create_mirror_view(d_y);
    Kokkos::View<float *>::HostMirror z = Kokkos::create_mirror_view(d_z);

    const float alpha = 2.0f;

    for (size_t idx = 0; idx < num_elements; idx++) {
      x(idx) = 1.0f;
      y(idx) = 2.0f;
      z(idx) = 0.0f;
    }

    Kokkos::deep_copy(d_x, x);
    Kokkos::deep_copy(d_y, y);
    Kokkos::deep_copy(d_z, z);

    Kokkos::parallel_for(
        "saxpy", num_elements,
        KOKKOS_LAMBDA(size_t idx) { d_z(idx) += alpha * d_x(idx) + d_y(idx); });

    Kokkos::deep_copy(z, d_z);

    float error = 0.0;
    for (size_t idx = 0; idx < num_elements; idx++) {
      error = fmax(error, fabs(z(idx) - 4.0f));
    }
    printf("error: %e (%s)\n", error, error == 0.0 ? "PASS" : "FAIL");
  }

  Kokkos::finalize();

  return 0;
}
