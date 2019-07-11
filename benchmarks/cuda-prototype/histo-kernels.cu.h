#ifndef HISTO_KERNELS
#define HISTO_KERNELS

#define STRIDED_MODE_LOC 1
#define STRIDED_MODE_GLB 0

enum AtomicPrim {ADD, CAS, XCHG};
enum MemoryType {GLBMEM, LOCMEM};

template<class T>
struct indval {
  int index;
  T value;
};

template<class T>
__device__ __host__ inline
struct indval<T>
f(T pixel, int his_sz) {
  const int ratio = max(1, his_sz/RACE_FACT);
  struct indval<T> iv;
  const int contraction = (((int)pixel) % ratio);
#if (CTGRACE || (STRIDE==1) || (RACE_FACT==1))
  iv.index = contraction;
#else
  iv.index = contraction * RACE_FACT;
#endif
  iv.value = pixel;
  return iv;
}

__global__ void
naive_reduce_kernel (int * d_his, int * d_res, int his_sz, int num_hists) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if(gid < his_sz) {
        for(int i=gid; i<num_hists * his_sz; i+=his_sz)
            sum += d_his[i];
        d_res[gid] = sum;
    }
}

/**********************************************/
/*** The three primitives for atomic update ***/
/**********************************************/
__device__ inline static void
atomADD(volatile int* loc_hists, volatile int* locks, int idx, int v) {
    atomicAdd((int*)&loc_hists[idx], v);
}
__device__ inline static void
atomCAS(volatile int* loc_hists, volatile int* locks, int idx, int v) {
    int old = loc_hists[idx];
    int assumed;
    do {
        assumed = old;
        old = atomicCAS( (int*)&loc_hists[idx], assumed, assumed + v );
    } while(assumed != old);
}
__device__ inline static void
atomXCGloc(volatile int* loc_hists, volatile int* loc_locks, int idx, int v) {
    bool done = false;
    while(!done) {
        if( atomicExch((int *)&loc_locks[idx], 1) == 0 ) {
            loc_hists[idx] += v;
            atomicExch((int *)&loc_locks[idx], 0);
            done = true;
        }
        __threadfence();
    }
}
__device__ inline static void
atomXCGglb(volatile int* loc_hists, volatile int* loc_locks, int idx, int v) {
    bool done = false;
    while(!done) {
        if( atomicExch((int *)&loc_locks[idx], 1) == 0 ) {
            loc_hists[idx] += v;
            __threadfence();
            loc_locks[idx] = 0;
            done = true;
        }
        __threadfence();
    }
}
// compile-time selector of atomic-update primitive
template<AtomicPrim primKind, MemoryType memKind> __device__ inline void
selectAtomicAdd(volatile int* loc_hists, volatile int* locks, int idx, int v) {
    if      (primKind == ADD)    atomADD(loc_hists, NULL,  idx, v);
    else if (primKind == CAS)    atomCAS(loc_hists, NULL,  idx, v);
    else if (memKind  == LOCMEM) atomXCGloc(loc_hists, locks, idx, v); // primKind == XCHG
    else                         atomXCGglb(loc_hists, locks, idx, v); // memKind  == GLBMEM                      
}

/**************************************************/
/***  Local-Memory Histogram Computation Kernel ***/
/**************************************************/
/**
 * Nomenclature:
 * N size of input array
 * H size of one histogram
 * M degree of sub-histogramming per block/workgroup
 *   (one workgroup computes M histograms)
 * C is the cooperation level ceil(BLOCK/M)
 * T the number of used hardware threads, i.e., T = min(N, Thwd_max)
 * histos: the global-memory array to store the subhistogram result. 
 */
template<AtomicPrim primKind>
__global__ void
locMemHwdAddCoopKernel0( const int N, const int H,
                        const int M, const int T,
                        int* input, int* histos
) {
    extern __shared__ volatile int loc_hists[];
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + tid;
    int his_block_sz = M * H;
    int ghid = blockIdx.x * M * H;

#if STRIDED_MODE_LOC
    int lhid = (tid % M) * H;
#else
    int C = (blockDim.x + M - 1) / M;
    int lhid = (tid / C) * H;    
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
    //if(gid < T) 
    {
        for(int i=gid; i<N; i+=T) {
            struct indval<int> iv = f<int>(input[i], H);
            selectAtomicAdd<primKind, LOCMEM>( loc_hists, loc_hists+his_block_sz
                                             , lhid+iv.index, iv.value );
        }
    }
    __syncthreads();

    // copy local histograms to global memory
    for(int i=tid; i<his_block_sz; i+=blockDim.x) {
        histos[ghid + i] = loc_hists[i];
    }
}

template<AtomicPrim primKind>
__global__ void
locMemHwdAddCoopKernel( const int N, const int H, const int M
                      , const int chunk_beg, const int chunk_end 
                      , const int T, int* input,  int* histos
) {
    extern __shared__ volatile int loc_hists[];
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + tid;
    const unsigned int Hchunk = chunk_end - chunk_beg;
    int his_block_sz = M * Hchunk;
    int ghid = blockIdx.x * M * H;
    

#if STRIDED_MODE_LOC
    int lhid = (tid % M) * Hchunk;
#else
    int C = (blockDim.x + M - 1) / M;
    int lhid = (tid / C) * Hchunk;    
#endif
    
    { // initialize local histograms (and locks if in case XCHG)
        unsigned int tot_len = M*(chunk_end - chunk_beg);
        if(primKind == XCHG)  tot_len *= 2;

        for(int i=tid; i<tot_len; i+=blockDim.x) {
            loc_hists[i] = 0;
        }
        __syncthreads();
    }

    // compute local histograms
    //if(gid < T) 
    {
        for(int i=gid; i<N; i+=T) {
          struct indval<int> iv = f<int>(input[i], H);
          if (iv.index >= chunk_beg && iv.index < chunk_end)
            selectAtomicAdd<primKind, LOCMEM>( loc_hists, loc_hists+his_block_sz
                                             , lhid+iv.index-chunk_beg, iv.value );
        }
    }
    __syncthreads();

    // copy local histograms to global memory
    for(int i=tid; i<his_block_sz; i+=blockDim.x) {
        const int hist_ind =  i / Hchunk;
        const int indH = (i % Hchunk) + chunk_beg;
        histos[ghid + hist_ind*H + indH] = loc_hists[i];
    }
}

/**************************************************/
/*** Global-Memory Histogram Computation Kernel ***/
/**************************************************/
template<AtomicPrim primKind>
__global__ void
glbMemHwdAddCoopKernel( const int N, const int H,
                        const int M, const int T,
                        const int chunk_beg, const int chunk_end,
                        int* input,
                        volatile int* histos,
                        volatile int* locks
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
#if STRIDED_MODE_GLB
    int ghidx = (gid % M) * H;
#else
    int C = (T + M - 1) / M;
    int ghidx = (gid / C) * H;
#endif
    // compute histograms; assumes histograms have been previously initialized
    for(int i=gid; i<N; i+=T) {
        struct indval<int> iv = f<int>(input[i], H);
        if (iv.index >= chunk_beg && iv.index < chunk_end)
            selectAtomicAdd<primKind, GLBMEM>(histos, locks, ghidx+iv.index, iv.value);
    }
}
#endif // HISTO_KERNELS
