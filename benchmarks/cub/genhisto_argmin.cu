#include "cub.cuh"   // or equivalently <cub/device/device_histogram.cuh>
#include "helper.cu.h"

#define real  uint64_t

__device__ __host__ inline static
uint64_t pack64(uint32_t ind, uint32_t val) {
   uint64_t res = ind;
   uint64_t tmp = val;
   tmp = tmp << 32;
   res = res | tmp;
   return res;
}

struct indval {
  uint32_t index;
  uint32_t value;
};

__device__ __host__ inline static
indval unpack64(uint64_t t) {
   const uint64_t MASK32bits = 4294967295;
   indval res;
   res.index = (uint32_t) (t & MASK32bits);
   res.value = (uint32_t) (t >> 32);
   return res;
}

__device__ __host__ inline static
uint64_t argmin(uint64_t v1, uint64_t v2) {
    indval arg1 = unpack64(v1);
    indval arg2 = unpack64(v2);
    uint32_t ind, val;
    if (arg1.value < arg2.value) {
        ind = arg2.index; val = arg2.value;
    } else if (arg1.value > arg2.value) {
        ind = arg1.index; val = arg1.value;
    } else { // arg1.value == arg2.value
        ind = min(arg1.index, arg2.index);
        val = arg1.value;
    }
    return pack64(ind, val);
}

void randomInit(uint32_t* keys, uint64_t* vals, const uint32_t N, const uint32_t H) {
    for (int i = 0; i < N; ++i) {
        uint32_t r = rand();
        uint32_t k = r % H;
        uint64_t v = pack64( (uint32_t)r/64, (uint32_t)r );
        keys[i] = k;
        vals[i] = v;
    }
}

void histoGold(uint32_t* keys, uint64_t* vals, const uint32_t N, const uint32_t H, uint64_t* histo) {
  for(uint32_t i = 0; i < H; i++) {
    histo[i] = 0.0;
  }
  for(int i = 0; i < N; i++) {
    uint32_t ind = keys[i];
    uint64_t v   = vals[i];
    histo[ind]   = argmin(histo[ind], v);
  }
}

bool validate(uint64_t* A, uint64_t* B, uint32_t H) {
    for(int i = 0; i < H; i++)
      if ( A[i] != B[i] ) {
        printf("INVALID RESULT %d (%lu,%lu)\n", i, A[i], B[i]);
        return false;
      }
    printf("VALID RESULT!\n");
    return true;
}

struct ArgMin
{
    __device__ CUB_RUNTIME_FUNCTION __forceinline__
    uint64_t operator()(const uint64_t &a, const uint64_t &b) const {
        return argmin(a, b);
    }
};

double sortRedByKeyCUB( uint32_t* data_keys_in,  uint64_t* data_vals_in
                      , uint64_t* histo, const uint32_t N, const uint32_t H
) {
    uint32_t* data_keys_out;
    uint64_t* data_vals_out;
    uint32_t* unique_keys;
    uint32_t* num_segments;

    int beg_bit = 0;
    int end_bit = ceilLog2(H);

    { // allocating stuff
        cudaMalloc ((void**) &data_keys_out, N * sizeof(uint32_t));
        cudaMalloc ((void**) &data_vals_out, N * sizeof(uint64_t));
        cudaMalloc ((void**) &unique_keys,   H * sizeof(uint32_t));
        cudaMalloc ((void**) &num_segments,  sizeof(uint32_t));
    }

    void * tmp_sort_mem = NULL;
    size_t tmp_sort_len = 0;

    { // sort prelude
        cub::DeviceRadixSort::SortPairs	( tmp_sort_mem, tmp_sort_len
                                        , data_keys_in, data_keys_out
                                        , data_vals_in, data_vals_out
                                        , (int)N,   beg_bit,  end_bit
                                    );
        cudaMalloc(&tmp_sort_mem, tmp_sort_len);
    }
    cudaCheckError();

    void * tmp_red_mem = NULL;
    size_t tmp_red_len = 0;
    ArgMin redop;

    { // reduce-by-key prelude
        cub::DeviceReduce::ReduceByKey  ( tmp_red_mem, tmp_red_len
                                        , data_keys_out, unique_keys
                                        , data_vals_out, histo
                                        , num_segments, redop, (int)N
                                        );
        cudaMalloc(&tmp_red_mem, tmp_red_len);
    }
    cudaCheckError();

    { // one dry run
        cub::DeviceRadixSort::SortPairs	( tmp_sort_mem, tmp_sort_len
                                        , data_keys_in, data_keys_out
                                        , data_vals_in, data_vals_out
                                        , (int)N,   beg_bit,  end_bit
                                        );
        cub::DeviceReduce::ReduceByKey  ( tmp_red_mem, tmp_red_len
                                        , data_keys_out, unique_keys
                                        , data_vals_out, histo
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
        cub::DeviceRadixSort::SortPairs ( tmp_sort_mem, tmp_sort_len
                                        , data_keys_in, data_keys_out
                                        , data_vals_in, data_vals_out
                                        , (int)N,   beg_bit,  end_bit
                                        );
        cub::DeviceReduce::ReduceByKey  ( tmp_red_mem, tmp_red_len
                                        , data_keys_out, unique_keys
                                        , data_vals_out, histo
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
    cudaFree(data_vals_out);
    cudaFree(unique_keys);
    cudaFree(num_segments);

    return elapsed;
}


int main (int argc, char * argv[]) {
    if (argc != 3) {
        printf("Usage: %s <histogram size> <image size>\n", argv[0]);
        exit(1);
    }
    const uint32_t N = atoi(argv[1]);
    const uint32_t H = atoi(argv[2]);
    printf("Computing for image size: %d and histogram size: %d\n", N, H);

    //Allocate and Initialize Host data with random values
    uint32_t* h_keys  = (uint32_t*) malloc(N*sizeof(uint32_t));
    uint64_t* h_vals  = (uint64_t*) malloc(N*sizeof(uint64_t));
    uint64_t* h_histo = (uint64_t*) malloc(H*sizeof(uint64_t));
    uint64_t* g_histo = (uint64_t*) malloc(H*sizeof(uint64_t));
    randomInit(h_keys, h_vals, N, H);

    { // golden sequential histogram
        double elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        histoGold(h_keys, h_vals, N, H, g_histo);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
        printf("Golden (Sequential) Key-Uint64 ArgMin Histogram runs in: %.2f microsecs\n", elapsed);
    }

    //Allocate and Initialize Device data
    uint32_t* d_keys;
    uint64_t* d_vals;
    uint64_t* d_histo;
    cudaMalloc ((void**) &d_keys,  N * sizeof(uint32_t));
    cudaMalloc ((void**) &d_vals,  N * sizeof(uint64_t));
    cudaMalloc ((void**) &d_histo, H * sizeof(uint64_t));
    cudaMemcpy(d_keys, h_keys, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, h_vals, N * sizeof(uint64_t), cudaMemcpyHostToDevice);

    double elapsed = 
      sortRedByKeyCUB ( d_keys,  d_vals, d_histo, N, H );

    cudaMemcpy(h_histo, d_histo, H*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaCheckError();
    printf("CUB Key-Uint64 ArgMin Histogram ... ");
    bool success = validate(g_histo, h_histo, H);

    printf("CUB Key-Uint64 ArgMin Histogram runs in: %.2f microsecs\n", elapsed);
    double gigaBytesPerSec = N * (sizeof(uint32_t) + 3*sizeof(uint64_t)) * 1.0e-3f / elapsed; 
    printf("CUB Key-Uint64 ArgMin Histogram GBytes/sec = %.2f!\n", gigaBytesPerSec); 

    // Cleanup and closing
    cudaFree(d_keys); cudaFree(d_vals); cudaFree(d_histo);
    free(h_keys);  free(h_vals); free(g_histo); free(h_histo);

    return success ? 0 : 1;
}
