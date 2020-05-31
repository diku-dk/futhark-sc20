#include "cub.cuh"  // or equivalently <cub/device/device_histogram.cuh>
#include "helper.cu.h"

template<class Z>
bool validateZ(Z* A, Z* B, uint32_t sizeAB) {
    for(uint32_t i = 0; i < sizeAB; i++)
      if (A[i] != B[i]){
        printf("INVALID RESULT %d (%d, %d)\n", i, A[i], B[i]);
        return false;
      }
    return true;
}

void randomInitNat(uint32_t* data, const uint32_t size, const uint32_t H) {
    for (int i = 0; i < size; ++i) {
        unsigned long int r = rand();
        data[i] = r % H;
    }
}

void histoGold(uint32_t* data, const uint32_t len, const uint32_t H, uint32_t* histo) {
  for(uint32_t i = 0; i < H; i++) {
    histo[i] = 0;
  }
  for(int i = 0; i < len; i++) {
    uint32_t ind = data[i];
    histo[ind]++;
  }
}

int main (int argc, char * argv[]) {
    if (argc != 3 && argc != 4) {
        printf("Usage: %s <N> <H> [timing-file]\n", argv[0]);
        exit(1);
    }
    const uint32_t N = atoi(argv[1]);
    const uint32_t H = atoi(argv[2]);

    //Allocate and Initialize Host data with random values
    uint32_t* h_data  = (uint32_t*)malloc(N*sizeof(uint32_t));
    uint32_t* h_histo = (uint32_t*)malloc(H*sizeof(uint32_t));
    uint32_t* g_histo = (uint32_t*)malloc(H*sizeof(uint32_t));
    randomInitNat(h_data, N, H);

    histoGold(h_data, N, H, g_histo);

    //Allocate and Initialize Device data
    uint32_t* d_data;
    uint32_t* d_histo;
    cudaSucceeded(cudaMalloc((void**) &d_data,  N * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &d_histo, H * sizeof(uint32_t)));
    cudaSucceeded(cudaMemcpy(d_data, h_data, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // CUB histogram version

    // Determine temporary device storage requirements
    cub::DeviceHistogram::HistogramEven( d_temp_storage, temp_storage_bytes
                                       , d_data, d_histo, H+1, (uint32_t)0
                                       , H, (int32_t)N);
    cudaCheckError();

    // Allocate temporary storage
    cudaSucceeded(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    { // one dry run
        cub::DeviceHistogram::HistogramEven( d_temp_storage, temp_storage_bytes
                                           , d_data, d_histo, H+1, (uint32_t)0
                                           , H, (int32_t)N );
        cudaDeviceSynchronize();
    }
    cudaCheckError();

    double elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    // Compute histogram: excluding inspector time
    for (uint32_t k=0; k<GPU_RUNS; k++) {
        cub::DeviceHistogram::HistogramEven( d_temp_storage, temp_storage_bytes
                                           , d_data, d_histo, H+1, (uint32_t)0
                                           , H, (int32_t)N );
    }
    cudaDeviceSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)GPU_RUNS);

    cudaCheckError();

    cudaSucceeded(cudaMemcpy (h_histo, d_histo, H*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    bool success = validateZ<uint32_t>(g_histo, h_histo, H);

    printf("CUB HDW histogram for N=%d, H=%d runs in: %.2f us\n", N, H, elapsed);

    if (argc == 4) {
      writeRuntime(argv[3], elapsed);
    }

    // Cleanup and closing
    cudaFree(d_data); cudaFree(d_histo); cudaFree(d_temp_storage);
    free(h_data); free(g_histo); free(h_histo);

    return success ? 0 : 1;
}
