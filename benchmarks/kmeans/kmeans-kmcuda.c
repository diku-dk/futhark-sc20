#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <kmcuda.h>

#include <sys/time.h>
static struct timeval t_start, t_end;

// ./example /path/to/data <number of clusters>
int main(int argc, const char **argv) {
  assert(argc == 3);
  // we open the binary file with the data (in Futhark binary format)
  // k
  // [samples_size][features_size][samples_size x features_size]
  FILE *fin = fopen(argv[1], "rb");
  assert(fin);
  uint64_t samples_size, features_size;

  int clusters_size;
  // Skip header.
  fseek(fin, 1 + 1 + 1 + 4, SEEK_CUR);
  assert(fread(&clusters_size, sizeof(clusters_size), 1, fin) == 1);

  // Skip header.
  fseek(fin, 1 + 1 + 1 + 4, SEEK_CUR);
  assert(fread(&samples_size, sizeof(samples_size), 1, fin) == 1);
  assert(fread(&features_size, sizeof(features_size), 1, fin) == 1);

  uint64_t total_size = ((uint64_t)samples_size) * features_size;
  float *samples = malloc(total_size * sizeof(float));
  assert(samples != NULL);
  assert(fread(samples, sizeof(float), total_size, fin) == total_size);
  fclose(fin);
  // we will store cluster centers here
  float *centroids = malloc(clusters_size * features_size * sizeof(float));
  assert(centroids);
  // we will store assignments of every sample here
  uint32_t *assignments = malloc(((uint64_t)samples_size) * sizeof(uint32_t));
  assert(assignments);
  float average_distance;

  int runs = 10;

  printf("Benchmarking kmcuda for %d runs with n=%d; d=%d; k=%d\n",
         runs,
         samples_size,
         features_size,
         clusters_size);

  KMCUDAResult result;
  for (int i = -1; i < runs; i++) {
    if (i == 0) // iter -1 is a warmup run.
      gettimeofday(&t_start, NULL);
    result = kmeans_cuda(kmcudaInitMethodPlusPlus, NULL,  // kmeans++ centroids initialization
                         0.01,                            // less than 1% of the samples are reassigned in the end
                         0.1,                             // activate Yinyang refinement with 0.1 threshold
                         kmcudaDistanceMetricL2,          // Euclidean distance
                         samples_size, features_size, clusters_size,
                         0xDEADBEEF,                      // random generator seed
                         1,                               // use only first device
                         -1,                              // samples are supplied from host
                         0,                               // not in float16x2 mode
                         0,                               // no verbosity
                         samples, centroids, assignments, &average_distance);
    assert(result == kmcudaSuccess);
  }

  gettimeofday(&t_end, NULL);

  free(samples);
  free(centroids);
  free(assignments);
  double elapsed_seconds = ((t_end.tv_sec+t_end.tv_usec/1e6) - (t_start.tv_sec+t_start.tv_usec/1e6)) / runs;
  printf("Average runtime: %fs (also written to %s)\n", elapsed_seconds, argv[2]);

  FILE *f = fopen(argv[2], "w");
  assert(f != NULL);
  fprintf(f, "%f\n", elapsed_seconds);
  fclose(f);

  return 0;
}
