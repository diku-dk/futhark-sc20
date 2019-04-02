#include "cub.cuh"   // or equivalently <cub/device/device_histogram.cuh>

#define GPU_RUNS    50
#define real double

#include "helper-keyValf64Add.cu.h"

struct RealAdd
{
    template <typename T>
    __device__ CUB_RUNTIME_FUNCTION __forceinline__
    T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

double sortRedByKeyCUB( uint32_t* data_keys_in,  real* data_vals_in
                      , real* histo, const uint32_t N, const uint32_t H
) {
    uint32_t* data_keys_out;
    real*     data_vals_out;
    uint32_t* unique_keys;
    uint32_t* num_segments;

    { // allocating stuff
        cudaMalloc ((void**) &data_keys_out, N * sizeof(uint32_t));
        cudaMalloc ((void**) &data_vals_out, N * sizeof(real));
        cudaMalloc ((void**) &unique_keys,   H * sizeof(uint32_t));
        cudaMalloc ((void**) &num_segments,  sizeof(uint32_t));
    }

    void * tmp_sort_mem = NULL;
    size_t tmp_sort_len = 0;

    { // sort prelude
        cub::DeviceRadixSort::SortPairs	( tmp_sort_mem, tmp_sort_len
                                        , data_keys_in, data_keys_out
                                        , data_vals_in, data_vals_out
                                        , (int)N
                                    );
        cudaMalloc(&tmp_sort_mem, tmp_sort_len);
    }

    void * tmp_red_mem = NULL;
    size_t tmp_red_len = 0;
    RealAdd redop;
    

    { // reduce-by-key prelude
        cub::DeviceReduce::ReduceByKey  ( tmp_red_mem, tmp_red_len
                                        , data_keys_out, unique_keys
                                        , data_vals_out, histo
                                        , num_segments, redop, (int)N
                                        );
        cudaMalloc(&tmp_red_mem, tmp_red_len);
    }

    { // one dry run
        cub::DeviceRadixSort::SortPairs	( tmp_sort_mem, tmp_sort_len
                                        , data_keys_in, data_keys_out
                                        , data_vals_in, data_vals_out
                                        , (int)N
                                        );
        cub::DeviceReduce::ReduceByKey  ( tmp_red_mem, tmp_red_len
                                        , data_keys_out, unique_keys
                                        , data_vals_out, histo
                                        , num_segments, redop, (int)N
                                        );
        cudaThreadSynchronize();
    }

    // timing
    double elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 

    for(int k=0; k<GPU_RUNS; k++) {
        cub::DeviceRadixSort::SortPairs ( tmp_sort_mem, tmp_sort_len
                                        , data_keys_in, data_keys_out
                                        , data_vals_in, data_vals_out
                                        , (int)N
                                        );
        cub::DeviceReduce::ReduceByKey  ( tmp_red_mem, tmp_red_len
                                        , data_keys_out, unique_keys
                                        , data_vals_out, histo
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
    cudaFree(data_vals_out);
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
    real*    h_vals  = (real*)   malloc(N*sizeof(real));
    real*    h_histo = (real*)   malloc(H*sizeof(real));
    real*    g_histo = (real*)   malloc(H*sizeof(real));
    randomInit(h_keys, h_vals, N, H);

    { // golden sequential histogram
        double elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 

        histoGold(h_keys, h_vals, N, H, g_histo);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
        printf("Golden (Sequential) Key-RealValue Add Histogram runs in: %.2f microsecs\n", elapsed);
    }

    //Allocate and Initialize Device data
    uint32_t* d_keys;
    real*    d_vals;
    real*    d_histo;
    cudaMalloc ((void**) &d_keys,  N * sizeof(uint32_t));
    cudaMalloc ((void**) &d_vals,  N * sizeof(real));
    cudaMalloc ((void**) &d_histo, H * sizeof(real));
    cudaMemcpy(d_keys, h_keys, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, h_vals, N * sizeof(real),    cudaMemcpyHostToDevice);

    {
        double elapsed = 
            sortRedByKeyCUB ( d_keys,  d_vals, d_histo, N, H );

        cudaMemcpy (h_histo, d_histo, H*sizeof(real), cudaMemcpyDeviceToHost);
        printf("CUB Key-RealValue Add Histogram ... ");
        validate(g_histo, h_histo, H);

        printf("CUB Key-RealValue Add Histogram runs in: %.2f microsecs\n", elapsed);
        double gigaBytesPerSec = N * (sizeof(uint32_t) + 3*sizeof(real)) * 1.0e-3f / elapsed; 
        printf("CUB Key-RealValue Add Histogram GBytes/sec = %.2f!\n", gigaBytesPerSec); 
    }

    // Cleanup and closing
    cudaFree(d_keys); cudaFree(d_vals); cudaFree(d_histo);
    free(h_keys);  free(h_vals); free(g_histo); free(h_histo);

    return 0;
}
