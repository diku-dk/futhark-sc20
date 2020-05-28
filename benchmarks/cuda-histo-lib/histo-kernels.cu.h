#ifndef HISTO_KERNELS
#define HISTO_KERNELS

#define STRIDED_MODE_LOC 1
#define STRIDED_MODE_GLB 0

enum AtomicPrim {HWD, CAS, XCG};
//enum MemoryType {GLBMEM, LOCMEM};

typedef unsigned int uint32_t;
//typedef unsigned long long int uint64_t;

template<class T>
struct indval {
  uint32_t index;
  T value;
};

/**************************************************/
/*** The three primitives for atomic update     ***/
/*** AtomicAdd demonstrated on int32_t addition ***/
/**************************************************/
__device__ inline static uint32_t
atomADDi32(volatile uint32_t* loc_hists, volatile int* locks, int idx, uint32_t v) {
    return atomicAdd((uint32_t*)&loc_hists[idx], v);
}

#if 1
// WHY doesn't this compiles???
__device__ inline static float
atomADDf32(volatile float* loc_hists, volatile int* locks, int idx, float v) {
    return atomicAdd((float*)&loc_hists[idx], v);
}
#endif

/*******************************************************************/
/*** CAS implementation demonstrated on a uint32_t saturated add ***/
/*******************************************************************/
template<class T>
__device__ inline static typename T::BETA
atomCAS32bit(volatile typename T::BETA* loc_hists, volatile int* locks, uint32_t idx, typename T::BETA v) {
    typedef typename T::BETA BETA;
    union { int32_t i; BETA f; } old;
    union { int32_t i; BETA f; } assumed;
    old.f = loc_hists[idx];
    do {
        assumed.f = old.f;
        old.f = T::opScal(assumed.f, v);
        old.i = atomicCAS( (int32_t*)&loc_hists[idx], assumed.i, old.i );
    } while(assumed.i != old.i);
    return old.f;
}

/*******************************************************************/
/*** Lock-Based Implementation demonstrated with ArgMin operator ***/
/*** the index and value are uint32_t and are packed in uint64_t ***/ 
/*******************************************************************/

template<class T>
__device__ inline static void
atomXCG(volatile typename T::BETA* loc_hists, volatile int* loc_locks, uint32_t idx, typename T::BETA v) {
    bool done = false;
    while(!done) {
        if( atomicExch((int *)&loc_locks[idx], 1) == 0 ) {
            loc_hists[idx] = T::opScal(loc_hists[idx], v);
            __threadfence();
            loc_locks[idx] = 0;
            done = true;
        }
        __threadfence();
    }
}

/************************************************************/
/*** Kernels for reducing across histograms (final stage) ***/
/************************************************************/
template<class T>
__global__ void
glbhist_reduce_kernel(typename T::BETA* d_his, typename T::BETA* d_res, int32_t his_sz, int32_t num_hists) {
    typedef typename T::BETA BETA;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < his_sz) {
        BETA sum = d_his[gid];
        for(int i = gid+his_sz; i < num_hists*his_sz; i+=his_sz)
            sum = T::opScal(sum, d_his[i]);
        d_res[gid] = sum;
    }
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
template<class HP>
__global__ void
locMemHwdAddCoopKernel( const int N, const int H
                      , const int M, const int T
                      , const int chunk_beg, const int chunk_end
                      , typename HP::ALPHA* input
                      , typename HP::BETA* histos
) {
    typedef typename HP::BETA BETA;

    extern __shared__ volatile char* loc_mem[];
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + tid;
    const unsigned int Hchunk = chunk_end - chunk_beg;
    unsigned int his_block_sz = M * Hchunk;
    volatile BETA* loc_hists =  (volatile BETA*) loc_mem;
    volatile int*  loc_locks =  (HP::atomicKind() != XCG) ? NULL :
                                //(volatile int*) ( (&loc_mem[0]) + sizeof(BETA)*his_block_sz); // BUG???? why doesn't it work???
                                (volatile int*) (loc_hists + his_block_sz);

#if STRIDED_MODE_LOC
    int lhid = (tid % M) * Hchunk;
#else
    int C = (blockDim.x + M - 1) / M;
    int lhid = (tid / C) * Hchunk;    
#endif
    
    { // initialize local histograms (and locks if in case XCG)
        for(int i=tid; i<his_block_sz; i+=blockDim.x) {
            loc_hists[i] = HP::ne();
        }
        if(HP::atomicKind() == XCG) {
            for(int i=tid; i<his_block_sz; i+=blockDim.x) {
                loc_locks[i] = 0;
            }
        }
        __syncthreads();
    }

    // compute local histograms
    //if(gid < T) 
    {
        // Loop was normalized so one can unroll
        int loop_count = (N - gid + T - 1) / T;
        for(int k=0; k<loop_count; k++) {
          int i = gid + k*T;
          struct indval<BETA> iv = HP::f(H, input[i]);
          if (iv.index >= chunk_beg && iv.index < chunk_end)
            HP::opAtom(loc_hists, loc_locks, lhid+iv.index-chunk_beg, iv.value);
        }
    }
    __syncthreads();

    // naive reduction of the histograms of the current block
    unsigned int upbd = M*Hchunk;
    for(int i = tid; (i < Hchunk) && (chunk_beg+i < H); i+=blockDim.x) {
        BETA acc = loc_hists[i];
        for(int j=Hchunk; j<upbd; j+=Hchunk) {
            BETA cur = loc_hists[i+j];
            acc = HP::opScal(acc, cur);
        }
        histos[blockIdx.x * H + chunk_beg + i] = acc;
    }
}

/**************************************************/
/*** Global-Memory Histogram Computation Kernel ***/
/**************************************************/
template<class HP>
__global__ void
glbMemHwdAddCoopKernel( const int N, const int H,
                        const int M, const int T,
                        const int chunk_beg, const int chunk_end,
                        typename HP::ALPHA* input,
                        volatile typename HP::BETA* histos,
                        volatile int*  locks
) {
    typedef typename HP::BETA BETA;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
#if STRIDED_MODE_GLB
    int ghidx = (gid % M) * H;
#else
    int C = (T + M - 1) / M;
    int ghidx = (gid / C) * H;
#endif
    // compute histograms; assumes histograms have been previously initialized
    for(int i=gid; i<N; i+=T) {
        struct indval<BETA> iv = HP::f(H, input[i]);
        if (iv.index >= chunk_beg && iv.index < chunk_end)
            HP::opAtom(histos, locks, ghidx+iv.index, iv.value);
    }
}

#endif // HISTO_KERNELS