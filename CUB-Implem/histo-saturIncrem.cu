#include "../../cub-1.8.0/cub/cub.cuh"   // or equivalently <cub/device/device_histogram.cuh>
#include "helper.cu.h"

#define GPU_RUNS    50

struct SatAdd
{
    template <typename T>
    __device__ CUB_RUNTIME_FUNCTION __forceinline__
    T operator()(const T &a, const T &b) const {
        uint32_t s = ((uint32_t)a) + ((uint32_t)b);
        return (s > 255) ? 255 : (T)s;
    }
};

void histoGold(uint32_t* vals, const uint32_t N, const uint32_t H, uint8_t* histo) {
  SatAdd satadd;
  for(uint32_t i = 0; i < H; i++) {
    histo[i] = 0;
  }
  for(int i = 0; i < N; i++) {
    uint32_t ind = vals[i];
    histo[ind]  = satadd(histo[ind], (uint8_t)1);
  }
}

double sortRedByKeyCUB( uint32_t* data_keys_in, uint8_t* histo
                      , const uint32_t N, const uint32_t H
) {
    uint32_t* data_keys_out;
    uint8_t * data_vals;
    uint32_t* unique_keys;
    uint32_t* num_segments;

    { // allocating stuff
        cudaMalloc ((void**) &data_keys_out, N * sizeof(uint32_t));
        cudaMalloc ((void**) &unique_keys,   H * sizeof(uint32_t));
        cudaMalloc ((void**) &num_segments,  sizeof(uint32_t));
        cudaMalloc ((void**) &data_vals,     N * sizeof(uint8_t));
        cudaMemset(data_vals, 1, N);
    }

    void * tmp_sort_mem = NULL;
    size_t tmp_sort_len = 0;

    { // sort prelude
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , (int)N
                                      );
        cudaMalloc(&tmp_sort_mem, tmp_sort_len);
    }

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

    { // one dry run
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , (int)N
                                      );
        cub::DeviceReduce::ReduceByKey( tmp_red_mem, tmp_red_len
                                      , data_keys_out, unique_keys
                                      , data_vals,     histo
                                      , num_segments, redop, (int)N
                                      );
        cudaThreadSynchronize();
    }

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
                                      , (int)N
                                      );
        cub::DeviceReduce::ReduceByKey( tmp_red_mem, tmp_red_len
                                      , data_keys_out, unique_keys
                                      , data_vals,     histo
                                      , num_segments, redop, (int)N
                                      );
    }
    cudaThreadSynchronize();

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
    if(argc != 3) {
        printf("Expects two arguments: the image size and the histogram size! argc:%d\n", argc);
        exit(1);
    }
    const uint32_t N = atoi(argv[1]);
    const uint32_t H = atoi(argv[2]);
    printf("Computing for image size: %d and histogram size: %d\n", N, H);

    //Allocate and Initialize Host data with random values
    uint32_t* h_keys  = (uint32_t*)malloc(N*sizeof(uint32_t));
    uint8_t * h_histo = (uint8_t*) malloc(H*sizeof(uint8_t ));
    uint8_t * g_histo = (uint8_t*) malloc(H*sizeof(uint8_t ));
    randomInitNat(h_keys, N, H);

    { // golden sequential histogram
        double elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        histoGold(h_keys, N, H, g_histo);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
        printf("Golden (Sequential) Uint8-Saturated-Add Histogram runs in: %.2f microsecs\n", elapsed);
    }

    //Allocate and Initialize Device data
    uint32_t* d_keys;
    uint8_t*  d_histo;
    cudaMalloc ((void**) &d_keys,  N * sizeof(uint32_t));
    cudaMalloc ((void**) &d_histo, H * sizeof(uint8_t));
    cudaMemcpy(d_keys, h_keys, N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    {
        double elapsed = 
            sortRedByKeyCUB ( d_keys, d_histo, N, H );

        cudaMemcpy (h_histo, d_histo, H*sizeof(uint8_t), cudaMemcpyDeviceToHost);
        printf("CUB Uint8-Saturated-Add Histogram ... ");
        validateZ<uint8_t>(g_histo, h_histo, H);

        printf("CUB Uint8-Saturated-Add Histogram runs in: %.2f microsecs\n", elapsed);
        double gigaBytesPerSec = N * (sizeof(uint32_t) + 2*sizeof(uint8_t)) * 1.0e-3f / elapsed; 
        printf("CUB Uint8-Saturated-Add Histogram GBytes/sec = %.2f!\n", gigaBytesPerSec); 
    }

    // Cleanup and closing
    cudaFree(d_keys); cudaFree(d_histo);
    free(h_keys); free(g_histo); free(h_histo);

    return 0;
}
