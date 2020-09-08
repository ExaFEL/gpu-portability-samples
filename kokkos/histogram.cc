#include <Kokkos_Core.hpp>

#define NUM_BUCKETS 128

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);

  size_t num_elements = 1 << 20;

  {
    Kokkos::View<float *> d_data("data", num_elements);
    Kokkos::View<unsigned *> d_histogram("histogram", NUM_BUCKETS);

    Kokkos::View<float *>::HostMirror data = Kokkos::create_mirror_view(d_data);
    Kokkos::View<unsigned *>::HostMirror histogram =
        Kokkos::create_mirror_view(d_histogram);

    float range = (float)RAND_MAX;
    for (size_t idx = 0; idx < num_elements; idx++) {
      data(idx) = rand();
    }
    for (size_t idx = 0; idx < NUM_BUCKETS; idx++) {
      histogram(idx) = 0;
    }

    Kokkos::deep_copy(d_data, data);
    Kokkos::deep_copy(d_histogram, histogram);

    typedef Kokkos::DefaultExecutionSpace::scratch_memory_space ScratchSpace;
    typedef Kokkos::View<unsigned *, ScratchSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        ScratchView;

    size_t elts_per_thread = 16;

    typedef Kokkos::TeamPolicy<>::member_type member_type;
    Kokkos::TeamPolicy<> policy(num_elements / elts_per_thread, Kokkos::AUTO());
    policy.set_scratch_size(
        0, Kokkos::PerTeam(ScratchView::shmem_size(NUM_BUCKETS)));
    Kokkos::parallel_for(
        "saxpy", policy, KOKKOS_LAMBDA(member_type mbr) {
          ScratchView local_histogram(mbr.team_scratch(0), NUM_BUCKETS);

          int t = mbr.team_rank();
          int nt = mbr.team_size();

          for (int i = t; i < NUM_BUCKETS; i += nt) local_histogram(i) = 0;

          mbr.team_barrier();

          for (int idx = mbr.league_rank() * mbr.team_size() + mbr.team_rank();
               idx < num_elements; idx += mbr.league_size() * mbr.team_size()) {
            size_t bucket = floor(data(idx) / range * (NUM_BUCKETS - 1));
            Kokkos::atomic_increment(&local_histogram(bucket));
          }

          mbr.team_barrier();

          for (int i = t; i < NUM_BUCKETS; i += nt)
            Kokkos::atomic_add(&d_histogram(i), local_histogram(i));
        });

    Kokkos::deep_copy(histogram, d_histogram);

    size_t total = 0;
    for (size_t idx = 0; idx < NUM_BUCKETS; idx++) {
      total += histogram(idx);
      printf("histogram[%lu] = %u\n", idx, histogram(idx));
    }
    printf("\ntotal = %lu (%s)\n", total,
           total == num_elements ? "PASS" : "FAIL");
  }

  Kokkos::finalize();

  return 0;
}
