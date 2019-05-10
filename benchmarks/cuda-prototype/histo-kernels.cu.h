#ifndef HISTO_KERNELS
#define HISTO_KERNELS

enum AtomicPrim {ADD, CAS, XCHG};

template<class T>
struct indval {
  int index;
  T value;
};

template<class T>
__device__ __host__ inline
struct indval<T>
f(T pixel, int his_sz)
{
  const int ratio = max(1, his_sz/RACE_FACT);
  struct indval<T> iv;
  iv.index = ((int)pixel) % ratio;
  iv.value = pixel;
  return iv;
}

__global__ void
naive_reduce_kernel (
              int * d_his,
              int * d_res,
              int his_sz,
              int num_hists )
{
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // sum bins
    int sum = 0;
    if(gid < his_sz) {
        for(int i=gid; i<num_hists * his_sz; i+=his_sz) {
            sum += d_his[i];
        }
        d_res[gid] = sum;
    }
}

/**********************************************/
/*** The three primitives for atomic update ***/
/**********************************************/

class HWDAddPrim {
  public:
    __device__ inline static void
    atomicAddPrim(volatile int* loc_hists, volatile int* locks, int idx, int v) {
        atomicAdd((int*)&loc_hists[idx], v);
    }
};

class CASAddPrim {
  public:
    __device__ inline static void
    atomicAddPrim(volatile int* loc_hists, volatile int* locks, int idx, int v) {
        int old = loc_hists[idx];
        int assumed;
        do {
            assumed = old;
            old = atomicCAS( (int*)&loc_hists[idx]
                           , assumed
                           , assumed + v
                           );
        } while(assumed != old);
    }
};


class XCGAddPrim {
  public:
    __device__ inline static void
    atomicAddPrim(volatile int* loc_hists, volatile int* loc_locks, int idx, int v) {
        bool done = false;
        do {
            if( atomicExch((int *)&loc_locks[idx], 1) == 0 ) {
                loc_hists[idx] += v;
                atomicExch((int *)&loc_locks[idx], 0);
                //__threadfence();
                //loc_locks[lhid + iv.index] = 0;
                done = true;
            }
        } while (!done);
    }
};

/**********************************************/

template<AtomicPrim primKind>
__global__ void
locMemHwdAddCoopKernel( const int N, const int H,
                        const int hists_per_block, const int num_threads,
                        int* d_input, int* d_histos
) {
    extern __shared__ volatile int loc_hists[];
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + tid;
    int his_block_sz = hists_per_block * H;
    int ghid = blockIdx.x * hists_per_block * H;

#ifdef SLOW_MODE
    int coop = (blockDim.x + hists_per_block - 1) / hists_per_block;
    int lhid = (tid / coop) * H;
#else
    int lhid = (tid % hists_per_block) * H;
#endif
    
    { // initialize local histograms (and locks if in case XCHG)
        unsigned int tot_len = his_block_sz;
        if(primKind == XCHG)  tot_len *= 2;

        for(int i=tid; i<tot_len; i+=blockDim.x) {
            loc_hists[i] = 0;
        }
        __syncthreads();
    }

    // compute local histograms
    //if(gid < num_threads) 
    {
        for(int i=gid; i<N; i+=num_threads) {
            struct indval<int> iv = f<int>(d_input[i], H);

            if(primKind == ADD) {
                HWDAddPrim::atomicAddPrim(loc_hists, NULL, lhid + iv.index, iv.value);
            } else if (primKind == CAS) {
                CASAddPrim::atomicAddPrim(loc_hists, NULL, lhid + iv.index, iv.value);
            } else { // primKind == XCHG
                XCGAddPrim::atomicAddPrim(loc_hists, loc_hists + his_block_sz, lhid + iv.index, iv.value);
            }
        }
    }
    __syncthreads();

    // copy local histograms to global memory
    for(int i=tid; i<his_block_sz; i+=blockDim.x) {
        d_histos[ghid + i] = loc_hists[i];
    }
}

#endif // HISTO_KERNELS
