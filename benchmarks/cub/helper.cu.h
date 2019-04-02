#ifndef HISTO_HELPERS
#define HISTO_HELPERS

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

void randomInitNat(uint32_t* data, const uint32_t size, const uint32_t H) {
    for (int i = 0; i < size; ++i) {
        unsigned long int r = rand();
        data[i] = r % H;
    }
}

template<class Z>
bool validateZ(Z* A, Z* B, uint32_t sizeAB) {
    for(uint32_t i = 0; i < sizeAB; i++)
      if (A[i] != B[i]){
        printf("INVALID RESULT %d (%d, %d)\n", i, A[i], B[i]);
        return false;
      }
    printf("VALID RESULT!\n");
    return true;
}

#endif // HISTO_HELPERS
