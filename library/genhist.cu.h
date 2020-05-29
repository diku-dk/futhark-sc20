// Single-header library for computing generalized
// histograms on CUDA GPUs.
//
// See example.cu for an example of how to use it.  A short
// description follows.
//
// The main entry point is the two classes LocalMemoryGenHist and
// GlobalMemoryGenHist, which encapsulate the state (mostly memory
// allocations) for computing generalized histograms for a certain
// number of bins, input length, and histogram descriptor (all of
// which must be given at creation time).  The two classes then define
// a method 'exec' for actually computing a generalized histogram, and
// 'result' for obtaining the memory in which the histogram is stored.

#pragma once

#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <cassert>

namespace genhist {

enum AtomicPrim {HWD, CAS, XCG};

template<class T>
struct indval {
  uint32_t index;
  T value;
};

// The three primitives for atomic update
// AtomicAdd demonstrated on int32_t addition
__device__ inline static uint32_t
atomADDi32(volatile uint32_t* loc_hists, volatile int* locks, int idx, uint32_t v) {
  return atomicAdd((uint32_t*)&loc_hists[idx], v);
}

__device__ inline static float
atomADDf32(volatile float* loc_hists, volatile int* locks, int idx, float v) {
  return atomicAdd((float*)&loc_hists[idx], v);
}

// CAS implementation demonstrated on a uint32_t saturated add
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

// Lock-Based Implementation demonstrated with ArgMin operator
// the index and value are uint32_t and are packed in uint64_t
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

// Kernels for reducing across histograms (final stage)
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

// Local-Memory Histogram Computation Kernel
//
// Nomenclature:
// N size of input array
// H size of one histogram
// M degree of sub-histogramming per block/workgroup
//   (one workgroup computes M histograms)
// C is the cooperation level ceil(BLOCK/M)
// T the number of used hardware threads, i.e., T = min(N, Thwd_max)
// histos: the global-memory array to store the subhistogram result.
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
    (volatile int*) (loc_hists + his_block_sz);

  int lhid = (tid % M) * Hchunk;

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

// Global-Memory Histogram Computation Kernel
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
  int C = (T + M - 1) / M;
  int ghidx = (gid / C) * H;
  // compute histograms; assumes histograms have been previously initialized
  for(int i=gid; i<N; i+=T) {
    struct indval<BETA> iv = HP::f(H, input[i]);
    if (iv.index >= chunk_beg && iv.index < chunk_end)
      HP::opAtom(histos, locks, ghidx+iv.index, iv.value);
  }
}

template<class T>
inline void
reduceAcrossMultiHistos(uint32_t H, uint32_t M, uint32_t B, typename T::BETA* d_histos, typename T::BETA* d_histo) {
  // reduce across subhistograms
  const size_t num_blocks_red = (H + B - 1) / B;
  glbhist_reduce_kernel<T><<< num_blocks_red, B >>>(d_histos, d_histo, H, M);
}

struct GenHistConfig
{
  const float k_RF;
  const float L2Fract;
  const int L2Cache;
  const int CLelmsz; // how many elements fit on a L2 cache line
  const int sharedMemWordsPerThread;
  const int glb_k_min;
  const int gpu_id;
};

const GenHistConfig rtx2080{ 0.75, 0.4, 4096*1024, 16, 12, 2, 0 };

template<class HP>
class GenHist
{
public:
  GenHist(int gpu_id) {
    int32_t nDevices;
    cudaGetDeviceCount(&nDevices);

    if (gpu_id >= nDevices) {
      throw std::invalid_argument("gpu_id out of range");
    }

    cudaGetDeviceProperties(&gpu_props, gpu_id);
  }

  virtual void exec(typename HP::ALPHA* d_input) = 0;
  virtual const typename HP::BETA* result() const = 0;

protected:

  inline int numThreads(int n) const {
    return std::min(n, getHWD());
  }

  inline int32_t getHWD() const {
    return gpu_props.maxThreadsPerMultiProcessor * gpu_props.multiProcessorCount;
  }

  inline int32_t getMaxBlockSize() const {
    return gpu_props.maxThreadsPerBlock;
  }

  inline int32_t getSH_MEM_SZ() const {
    return gpu_props.sharedMemPerBlock;
  }

  cudaDeviceProp gpu_props;
};

template<class HP>
class LocalMemoryGenHist : public GenHist<HP>
{
public:
  LocalMemoryGenHist(GenHistConfig consts, int H, int N)
    : GenHist<HP>(consts.gpu_id), H(H), N(N), consts(consts) {
    typedef typename HP::BETA BETA;
    const AtomicPrim prim_kind = HP::atomicKind();
    const int32_t BLOCK = GenHist<HP>::gpu_props.maxThreadsPerBlock;

    const int32_t lmem = consts.sharedMemWordsPerThread * BLOCK * 4;
    num_blocks = (GenHist<HP>::numThreads(N) + BLOCK - 1) / BLOCK;
    const int32_t q_small = 2;
    const int32_t work_asymp_M_max = N / (q_small*num_blocks*H);

    const int32_t elms_per_block = (N + num_blocks - 1) / num_blocks;
    const int32_t el_size = sizeof(BETA) + ( (prim_kind==XCG) ? sizeof(int) : 0 );
    float m_prime = std::min( (lmem*1.0F / el_size), (float)elms_per_block ) / H;

    M = std::max(1, min( (int)floor(m_prime), BLOCK ) );
    M = std::min(M, work_asymp_M_max);
    assert(M > 0);

    const int32_t len = lmem / (el_size * M);
    num_chunks = (H + len - 1) / len;

    const size_t mem_size_histo  = H * sizeof(BETA);
    const size_t mem_size_histos = num_blocks * mem_size_histo;
    cudaMalloc((void**) &d_histos, mem_size_histos);
    cudaMalloc((void**) &d_histo,  mem_size_histo);
    cudaMemset(d_histo, 0, mem_size_histo);

    const int32_t Hchunk = (H + num_chunks - 1) / num_chunks;
    shmem_size = M * Hchunk * el_size;
  }

  ~LocalMemoryGenHist() {
    cudaFree(d_histos);
    cudaFree(d_histo);
  }

  void exec(typename HP::ALPHA* d_input) {
    typedef typename HP::BETA BETA;
    const int32_t BLOCK  = GenHist<HP>::gpu_props.maxThreadsPerBlock;
    const int32_t Hchunk = (H + num_chunks - 1) / num_chunks;

    const size_t mem_size_histo  = H * sizeof(BETA);
    const size_t mem_size_histos = num_blocks * mem_size_histo;

    cudaMemset(d_histos, 0, mem_size_histos);
    for(int k=0; k<num_chunks; k++) {
      const int32_t chunkLB = k*Hchunk;
      const int32_t chunkUB = min(H, (k+1)*Hchunk);

      locMemHwdAddCoopKernel<HP><<< num_blocks, BLOCK, shmem_size >>>
        (N, H, M, GenHist<HP>::numThreads(N), chunkLB, chunkUB, d_input, d_histos);
    }

    // reduce across histograms
    reduceAcrossMultiHistos<HP>(H, num_blocks, 256, d_histos, d_histo);
  }

  const typename HP::BETA* result() const {
    return d_histo;
  }

private:
  const GenHistConfig consts;
  int H, N, M, num_chunks, num_blocks;
  typename HP::BETA* d_histos;
  typename HP::BETA* d_histo;
  size_t shmem_size;
};

template<class HP>
class GlobalMemoryGenHist : public GenHist<HP>
{
public:
  GlobalMemoryGenHist(GenHistConfig consts, int B, int RF, int H, int N)
    : GenHist<HP>(consts.gpu_id), B(B), RF(RF), H(H), N(N), consts(consts) {
    const int32_t T = GenHist<HP>::numThreads(N);
    typedef typename HP::BETA BETA;
    const AtomicPrim prim_kind = HP::atomicKind();

    // For the computation of avg_size on XCG:
    //   In principle we average the size of the lock and of the element-type of histogram
    const int   avg_size= (prim_kind == XCG)? ( sizeof(BETA) + sizeof(int) )/2 : sizeof(BETA);
    const int   el_size = (prim_kind == XCG)? sizeof(BETA) + sizeof(int) : sizeof(BETA);
    const float optim_k_min = consts.glb_k_min;
    const int   q_small = 2;
    const int   work_asymp_M_max = N / (q_small*H);

    // first part
    float race_exp = max(1.0, (1.0 * consts.k_RF * RF) / ( (4.0*consts.CLelmsz) / avg_size) );
    float coop_min = std::min( (float)T, H/optim_k_min );
    const int Mdeg  = min(work_asymp_M_max, max(1, (int) (T / coop_min)));
    const int S_nom = Mdeg*H*avg_size; //el_size;  // diference: Futhark using avg_size instead of `el_size` here, and seems to do better!
    const int S_den = (int) (consts.L2Fract * consts.L2Cache * race_exp);
    num_chunks = (S_nom + S_den - 1) / S_den;
    const int H_chk = (int)ceil( H / num_chunks );

    // second part
    const float u = (prim_kind == HWD) ? 2.0 : 1.0;
    const float k_max= std::min( consts.L2Fract * ( (1.0F*consts.L2Cache) / el_size ) * race_exp, (float)N ) / T;
    const float coop = std::min( (float)T, (u * H_chk) / k_max );
    M = max( 1, (int)floor(T/coop) );

    const int32_t C = (T + M - 1) / M;
    assert((C > 0) && (C <= T));

    const size_t mem_size_histo  = H * sizeof(BETA);
    const size_t mem_size_histos = M * mem_size_histo;
    cudaMalloc((void**) &d_histos, mem_size_histos);
    cudaMalloc((void**) &d_histo,  mem_size_histo );
    cudaMemset(d_histo,  0, mem_size_histo );

    if (prim_kind == XCG) {
      const size_t mem_size_locks = M * H * sizeof(int32_t);
      cudaMalloc((void**) &d_locks, mem_size_locks);
      cudaMemset(d_locks,  0, mem_size_locks );
    } else {
      d_locks = NULL;
    }
  }

  ~GlobalMemoryGenHist() {
    cudaFree(d_histos);
    cudaFree(d_histo);
    cudaFree(d_locks);
  }

  void exec(typename HP::ALPHA* d_input) {
    typedef typename HP::BETA BETA;
    const int32_t T = GenHist<HP>::numThreads(N);
    const int32_t chunk_size = (H + num_chunks - 1) / num_chunks;
    const int32_t num_blocks = (T + B - 1) / B;

    const size_t mem_size_histos = M * H * sizeof(BETA);
    cudaMemset(d_histos, 0, mem_size_histos);

    // compute histogram
    for(int k=0; k<num_chunks; k++) {
      glbMemHwdAddCoopKernel<HP><<< num_blocks, B >>>
        (N, H, M, T, k*chunk_size, (k+1)*chunk_size, d_input, d_histos, d_locks);
    }
    // reduce across subhistograms
    reduceAcrossMultiHistos<HP>(H, M, B, d_histos, d_histo);
  }

  const typename HP::BETA* result() const {
    return d_histo;
  }

private:
  int RF, H, N, M, num_chunks, B;
  typename HP::BETA* d_histos;
  typename HP::BETA* d_histo;
  int32_t*           d_locks;
  const GenHistConfig consts;
};

}
