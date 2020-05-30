#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

int main(int argc, char** argv) {
  assert(argc == 3);
  int64_t num_samples = atoi(argv[1]);
  int64_t num_features = atoi(argv[2]);

  fputc('b', stdout);
  fputc(2, stdout);
  fputc(2, stdout);
  fputs(" f32", stdout);
  fwrite(&num_samples, sizeof(int64_t), 1, stdout);
  fwrite(&num_features, sizeof(int64_t), 1, stdout);

  for (int i = 0; i < num_samples; i++) {
    for (int j = 0; j < num_features; j++) {
      float d = ((float)rand())/((float)RAND_MAX);
      fwrite(&d, sizeof(float), 1, stdout);
    }
  }
}
