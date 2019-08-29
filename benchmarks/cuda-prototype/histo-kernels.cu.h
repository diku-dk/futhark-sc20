#ifndef HISTO_KERNELS
#define HISTO_KERNELS

#define STRIDED_MODE_LOC 1
#define STRIDED_MODE_GLB 0

enum AtomicPrim {ADD, CAS, XCHG};
enum MemoryType {GLBMEM, LOCMEM};

typedef unsigned int uint32_t;
typedef unsigned long long int uint64_t;

template<class T>
struct indval {
  uint32_t index;
  T value;
};

template<class T>
__device__ __host__ inline
struct indval<T>
f(int pixel, uint32_t his_sz) {
  const uint32_t ratio = max(1, his_sz/RACE_FACT);
  struct indval<T> iv;
  const uint32_t contraction = (((uint32_t)pixel) % ratio);
#if (CTGRACE || (STRIDE==1) || (RACE_FACT==1))
  iv.index = contraction;
#else
  iv.index = contraction * RACE_FACT;
#endif
  iv.value = (T)pixel;
  return iv;
}


/**************************************************/
/*** The three primitives for atomic update     ***/
/*** AtomicAdd demonstrated on int32_t addition ***/
/**************************************************/
__device__ inline static void
atomADD(volatile uint32_t* loc_hists, volatile int* locks, int idx, uint32_t v) {
    atomicAdd((uint32_t*)&loc_hists[idx], v);
}

/*******************************************************************/
/*** CAS implementation demonstrated on a uint32_t saturated add ***/
/*******************************************************************/
__device__ __host__ inline static uint32_t
satadd(uint32_t v1, uint32_t v2) {
    const uint32_t SAT_VAL32 = 4294967295;
    uint32_t res;
    if(SAT_VAL32 - v1 < v2) {
        res = SAT_VAL32;
    } else {
        res = v1 + v2;
    }
    return res;
}
__device__ inline static void
atomCAS(volatile uint32_t* loc_hists, volatile int* locks, uint32_t idx, uint32_t v) {
    int old = loc_hists[idx];
    int assumed, upd;
    do {
        assumed = old;
        upd = satadd(assumed, v);
        old = atomicCAS( (int*)&loc_hists[idx], assumed, upd );
    } while(assumed != old);
}

/*******************************************************************/
/*** Lock-Based Implementation demonstrated with ArgMin operator ***/
/*** the index and value are uint32_t and are packed in uint64_t ***/ 
/*******************************************************************/
__device__ __host__ inline static
uint64_t pack64(uint32_t ind, uint32_t val) {
   uint64_t res = ind;
   uint64_t tmp = val;
   tmp = tmp << 32;
   res = res | tmp;
   return res;
}

__device__ __host__ inline static
indval<uint32_t> unpack64(uint64_t t) {
   const uint64_t MASK32bits = 4294967295;
   indval<uint32_t> res;
   res.index = (uint32_t) (t & MASK32bits);
   res.value = (uint32_t) (t >> 32);
   return res;
}

__device__ __host__ inline static
uint64_t argmin(uint64_t v1, uint64_t v2) {
    indval<uint32_t> arg1 = unpack64(v1);
    indval<uint32_t> arg2 = unpack64(v2);
    uint32_t ind, val;
    if (arg1.value < arg2.value) {
        ind = arg2.index; val = arg2.value;
    } else if (arg1.value > arg2.value) {
        ind = arg1.index; val = arg2.value;
    } else { // arg1.value == arg2.value
        ind = min(arg1.index, arg2.index);
        val = arg1.value;
    }
    return pack64(ind, val);
}

__device__ inline static void
atomXCGloc(volatile uint64_t* loc_hists, volatile int* loc_locks, uint32_t idx, uint64_t v) {
    bool done = false;
    while(!done) {
        if( atomicExch((int *)&loc_locks[idx], 1) == 0 ) {
            //loc_hists[idx] += v;
            loc_hists[idx] = argmin(loc_hists[idx], v);
            atomicExch((int *)&loc_locks[idx], 0);
            done = true;
        }
        __threadfence();
    }
}
__device__ inline static void
atomXCGglb(volatile uint64_t* glb_hists, volatile int* loc_locks, uint32_t idx, uint64_t v) {
    bool done = false;
    while(!done) {
        if( atomicExch((int *)&loc_locks[idx], 1) == 0 ) {
            //glb_hists[idx] += v;
            glb_hists[idx] = argmin(glb_hists[idx], v);
            __threadfence();
            loc_locks[idx] = 0;
            done = true;
        }
        __threadfence();
    }
}
/////////////////////////////////////////////////////////////////

// compile-time selector of atomic-update primitive
template<AtomicPrim primKind, MemoryType memKind, class BETA> __device__ inline void
selectAtomicAdd(volatile BETA* loc_hists, volatile int* locks, uint32_t idx, BETA v) {
    if      (primKind == ADD)    atomADD((volatile uint32_t*)loc_hists, NULL,  idx, v);
    else if (primKind == CAS)    atomCAS((volatile uint32_t*)loc_hists, NULL,  idx, v);
    else if (memKind  == LOCMEM) atomXCGloc((volatile uint64_t*)loc_hists, locks, idx, v); // primKind == XCHG
    else                         atomXCGglb((volatile uint64_t*)loc_hists, locks, idx, v); // memKind  == GLBMEM                      
}

/************************************************************/
/*** Kernels for reducing across histograms (final stage) ***/
/************************************************************/
__global__ void
naive_atmadd_reduce_kernel (uint32_t * d_his, uint32_t * d_res, int his_sz, int num_hists) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t sum = 0;
    if(gid < his_sz) {
        for(int i = gid; i < num_hists*his_sz; i+=his_sz)
            sum += d_his[i];
        d_res[gid] = sum;
    }
}
__global__ void
naive_satadd_reduce_kernel (uint32_t * d_his, uint32_t * d_res, int his_sz, int num_hists) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t sum = 0;
    if(gid < his_sz) {
        for(int i = gid; i < num_hists*his_sz; i+=his_sz)
            sum = satadd(sum, d_his[i]);
        d_res[gid] = sum;
    }
}
__global__ void
naive_argmin_reduce_kernel (uint64_t * d_his, uint64_t * d_res, int his_sz, int num_hists) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < his_sz) {
        uint64_t sum = d_his[gid];
        for(int i = gid+his_sz; i < num_hists*his_sz; i+=his_sz)
            sum = argmin(sum, d_his[i]);
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
template<AtomicPrim primKind, class BETA>
__global__ void
locMemHwdAddCoopKernel( const int N, const int H, const int M
                      , const int chunk_beg, const int chunk_end 
                      , const int T, int* input,  BETA* histos
) {
    extern __shared__ volatile int32_t loc_mem[];
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + tid;
    const unsigned int Hchunk = chunk_end - chunk_beg;
    unsigned int his_block_sz = M * Hchunk;
    int ghid = blockIdx.x * M * H;
    volatile BETA* loc_hists = (__shared__ volatile BETA*) loc_mem;
    volatile int* loc_locks  = (primKind != XCHG) ? NULL : (loc_mem + 2*his_block_sz);

#if STRIDED_MODE_LOC
    int lhid = (tid % M) * Hchunk;
#else
    int C = (blockDim.x + M - 1) / M;
    int lhid = (tid / C) * Hchunk;    
#endif
    
    { // initialize local histograms (and locks if in case XCHG)
        unsigned int tot_len = (primKind != XCHG) ? his_block_sz : 3 * his_block_sz;
        for(int i=tid; i<tot_len; i+=blockDim.x) {
            loc_mem[i] = 0;
        }
        __syncthreads();
    }

    // compute local histograms
    //if(gid < T) 
    {
        for(int i=gid; i<N; i+=T) {
          struct indval<BETA> iv = f<BETA>(input[i], H);
          if (iv.index >= chunk_beg && iv.index < chunk_end)
            selectAtomicAdd<primKind, LOCMEM, BETA>
                ( loc_hists, loc_locks
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
template<AtomicPrim primKind, class BETA>
__global__ void
glbMemHwdAddCoopKernel( const int N, const int H,
                        const int M, const int T,
                        const int chunk_beg, const int chunk_end,
                        int* input,
                        volatile BETA* histos,
                        volatile int*  locks
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
        struct indval<BETA> iv = f<BETA>(input[i], H);
        if (iv.index >= chunk_beg && iv.index < chunk_end)
            selectAtomicAdd<primKind, GLBMEM, BETA>(histos, locks, ghidx+iv.index, iv.value);
    }
}
#endif // HISTO_KERNELS
