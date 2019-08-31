//#include "../../cub-1.8.0/cub/cub.cuh"   // or equivalently <cub/device/device_histogram.cuh>
#include "cub.cuh"
#include "helper.cu.h"

__global__ void
setOnesKernel (uint32_t * d_vals, uint32_t N) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < N) {
        d_vals[gid] = 1;
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

void randomInitNat(uint32_t* data, const uint32_t size, const uint32_t H) {
    for (int i = 0; i < size; ++i) {
        unsigned long int r = rand();
        data[i] = r % H;
    }
}

#define GPU_RUNS    50

struct SatAdd
{
    __device__ CUB_RUNTIME_FUNCTION __forceinline__
    uint32_t operator()(const uint32_t &v1, const uint32_t &v2) const {
        const uint32_t SAT_VAL24 = (1 << 24) - 1;
/*
        uint32_t res;
        if(SAT_VAL24 - v1 < v2) {
            res = SAT_VAL24;
        } else {
            res = v1 + v2;
        }
        return res;
*/
        uint32_t s = v1 + v2;
        return (s > SAT_VAL24) ? SAT_VAL24 : s;
    }
};

void histoGold(uint32_t* vals, const uint32_t N, const uint32_t H, uint32_t* histo) {
  SatAdd satadd;
  for(uint32_t i = 0; i < H; i++) {
    histo[i] = 0;
  }
  for(int i = 0; i < N; i++) {
    uint32_t ind = vals[i];
    histo[ind]  = satadd(histo[ind], (uint32_t)1);
  }
}

double sortRedByKeyCUB( uint32_t* data_keys_in, uint32_t* histo
                      , const uint32_t N, const uint32_t H
) {
    uint32_t* data_keys_out;
    uint32_t* data_vals;
    uint32_t* unique_keys;
    uint32_t* num_segments;

    int beg_bit = 0;
    int end_bit = ceilLog2(H);

    { // allocating stuff
        cudaMalloc ((void**) &data_keys_out, N * sizeof(uint32_t));
        cudaMalloc ((void**) &unique_keys,   H * sizeof(uint32_t));
        cudaMalloc ((void**) &num_segments,  sizeof(uint32_t));
        cudaMalloc ((void**) &data_vals,     N * sizeof(uint32_t));
        { // setting data_vals to ones
            const int block = 256;
            const int grid  = (N + block - 1) / block;
            setOnesKernel<<<grid, block>>>(data_vals, N);
        }
    }

    void * tmp_sort_mem = NULL;
    size_t tmp_sort_len = 0;

    { // sort prelude
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , (int)N,   beg_bit,  end_bit
                                      );
        cudaMalloc(&tmp_sort_mem, tmp_sort_len);
    }
    cudaCheckError();

    void * tmp_red_mem = NULL;
    size_t tmp_red_len = 0;
    SatAdd redop;

    { // reduce-by-key prelude
        cub::DeviceReduce::ReduceByKey  ( tmp_red_mem, tmp_red_len
                                        , data_keys_out, unique_keys
                                        , data_vals,     histo
                                        , num_segments, redop, (int)N
                                        );
        cudaMalloc(&tmp_red_mem, tmp_red_len);
    }
    cudaCheckError();

    { // one dry run
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , (int)N,   beg_bit,  end_bit
                                      );
        cub::DeviceReduce::ReduceByKey( tmp_red_mem, tmp_red_len
                                      , data_keys_out, unique_keys
                                      , data_vals,     histo
                                      , num_segments, redop, (int)N
                                      );
        cudaDeviceSynchronize();
    }
    cudaCheckError();

    // timing
    double elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 

    for(int k=0; k<GPU_RUNS; k++) {
#if 0
        void* tmpNULL = NULL;
        cub::DeviceReduce::ReduceByKey  ( tmpNULL, tmp_red_len
                                        , data_keys_out, unique_keys
                                        , data_vals,     histo
                                        , num_segments, redop, (int)N
                                        );
        cub::DeviceReduce::ReduceByKey  ( tmpNULL, tmp_red_len
                                        , data_keys_out, unique_keys
                                        , data_vals,     histo
                                        , num_segments, redop, (int)N
                                        );
#endif
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , (int)N,   beg_bit,  end_bit
                                      );
        cub::DeviceReduce::ReduceByKey( tmp_red_mem, tmp_red_len
                                      , data_keys_out, unique_keys
                                      , data_vals,     histo
                                      , num_segments, redop, (int)N
                                      );
    }
    cudaDeviceSynchronize();
    cudaCheckError();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)GPU_RUNS);

    cudaFree(tmp_sort_mem);
    cudaFree(tmp_red_mem);
    cudaFree(data_keys_out);
    cudaFree(data_vals);
    cudaFree(unique_keys); 
    cudaFree(num_segments);

    return elapsed;
}


int main (int argc, char * argv[]) {
    if (argc != 3) {
        printf("Usage: %s <image size> <histogram size>\n", argv[0]);
        exit(1);
    }
    const uint32_t N = atoi(argv[1]);
    const uint32_t H = atoi(argv[2]);
    printf("Computing for image size: %d and histogram size: %d\n", N, H);

    //Allocate and Initialize Host data with random values
    uint32_t* h_keys  = (uint32_t*) malloc(N*sizeof(uint32_t));
    uint32_t* h_histo = (uint32_t*) malloc(H*sizeof(uint32_t ));
    uint32_t* g_histo = (uint32_t*) malloc(H*sizeof(uint32_t ));
    randomInitNat(h_keys, N, H);

    { // golden sequential histogram
        double elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        histoGold(h_keys, N, H, g_histo);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
        printf("Golden (Sequential) Uint24-Saturated-Add Histogram runs in: %.2f microsecs\n", elapsed);
    }

    //Allocate and Initialize Device data
    uint32_t* d_keys;
    uint32_t* d_histo;
    cudaSucceeded(cudaMalloc((void**) &d_keys,  N * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &d_histo, H * sizeof(uint32_t)));
    cudaSucceeded(cudaMemcpy(d_keys, h_keys, N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    double elapsed = sortRedByKeyCUB( d_keys, d_histo, N, H );

    cudaMemcpy(h_histo, d_histo, H*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaCheckError();
    printf("CUB Uint24-Saturated-Add Histogram ... ");
    bool success = validateZ<uint32_t>(g_histo, h_histo, H);

    printf("CUB Uint24-Saturated-Add Histogram runs in: %.2f microsecs\n", elapsed);
    double gigaBytesPerSec = N * (sizeof(uint32_t) + 2*sizeof(uint32_t)) * 1.0e-3f / elapsed; 
    printf("CUB Uint24-Saturated-Add Histogram GBytes/sec = %.2f!\n", gigaBytesPerSec); 

    // Cleanup and closing
    cudaFree(d_keys); cudaFree(d_histo);
    free(h_keys); free(g_histo); free(h_histo);

    return success ? 0 : 1;
}
