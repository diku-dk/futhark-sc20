#ifndef REDBYKEY_HELPERS
#define REDBYKEY_HELPERS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

void randomInit(uint32_t* keys, real* vals, const uint32_t N, const uint32_t H) {
    for (int i = 0; i < N; ++i) {
        uint32_t r = rand();
        uint32_t k = r % H;
        real     v = ((real)r) / RAND_MAX;
        keys[i] = k;
        vals[i] = v;
    }
}

void histoGold(uint32_t* keys, real* vals, const uint32_t N, const uint32_t H, real* histo) {
  for(uint32_t i = 0; i < H; i++) {
    histo[i] = 0.0;
  }
  for(int i = 0; i < N; i++) {
    uint32_t ind = keys[i];
    real     v   = vals[i];
    histo[ind]  += v;
  }
}

bool validate(real* A, real* B, uint32_t H) {
    for(int i = 0; i < H; i++)
      if (fabs(A[i] - B[i]) > 0.0001){
        printf("INVALID RESULT %d (%f,%f)\n", i, A[i], B[i]);
        return false;
      }
    printf("VALID RESULT!\n");
    return true;
}

#endif // REDBYKEY_HELPERS
