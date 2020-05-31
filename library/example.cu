// This program uses the generalized histogram library to compute
// histograms for various histogram sizes and operators.  It validates
// the result as compared to a sequential implementation.

#include "genhist.cu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <errno.h>
#include <string.h>

#define GPU_RUNS    100
#define CPU_RUNS    1

#define INP_LEN     50000000
#define Hmax        4000000

#define RESET   "\033[0m"
#define BLACK   "\033[30m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"
#define BOLD    "\033[1m"
#define BOLDBLACK   "\033[1m\033[30m"
#define BOLDRED     "\033[1m\033[31m"
#define BOLDGREEN   "\033[1m\033[32m"
#define BOLDYELLOW  "\033[1m\033[33m"
#define BOLDBLUE    "\033[1m\033[34m"
#define BOLDMAGENTA "\033[1m\033[35m"
#define BOLDCYAN    "\033[1m\033[36m"
#define BOLDWHITE   "\033[1m\033[37m"

// Helpers

int gpuAssert(cudaError_t code) {
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    exit(33);
    return -1;
  }
  return 0;
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
  unsigned int resolution=1000000;
  long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
  result->tv_sec = diff / resolution;
  result->tv_usec = diff % resolution;
  return (diff<0);
}

void randomInit(int32_t* data, int size) {
  for (int32_t i = 0; i < size; ++i)
    data[i] = rand(); // (float)RAND_MAX;
}

template<class T>
void zeroOut(typename T::BETA* data, int size) {
  for (int i = 0; i < size; ++i)
    data[i] = T::ne();
}

template<int num_histos, int num_m_degs>
void printTextTab(const unsigned long runtimes[3][num_histos][num_m_degs],
                  const int histo_sizes[num_histos],
                  const int kms[num_m_degs],
                  const int RF) {
  for(int k=0; k<3; k++) {
    printf("\n\n");

    printf(BOLD "%s, RF=%d\n" RESET,
           k == 0 ? "HDW" :
           k == 1 ? "CAS" :
           "XCG",
           RF);

    for(int i = 0; i<num_histos; i++) {
      if (histo_sizes[i] > 1000) {
        printf(BOLD "\tH=%dK" RESET, histo_sizes[i]/1000);
      } else {
        printf(BOLD "\tH=%d" RESET, histo_sizes[i]);
      }
    }

    printf("\n");

    for(int j=0; j<num_m_degs; j++) {
      if (j < num_m_degs-1) {
        printf(BOLD "M=%d\t" RESET, kms[j]);
      } else {
        printf(BOLD "Auto\t" RESET);
      }
      for(int i = 0; i<num_histos; i++) {
        printf("%lu\t", runtimes[k][i][j]);
      }
      printf("\n");
    }
  }
}

// Histogram descriptors

template<int RF>
struct AddI32 {
  typedef int32_t  BETA;
  typedef int32_t  ALPHA;

  __device__ __host__ inline static
  genhist::indval<BETA> f(const int32_t H, ALPHA pixel) {
    genhist::indval<BETA> res;
    const uint32_t ratio = max(1, H/RF);
    const uint32_t contraction = (((uint32_t)pixel) % ratio);
    res.index = contraction * RF;
    res.value = pixel;
    return res;
  }

  __device__ __host__ inline static
  BETA ne() { return 0; }

  __device__ __host__ inline static
  BETA opScal(BETA v1, BETA v2) {
    return v1 + v2;
  }

  __device__ __host__ inline static
  genhist::AtomicPrim atomicKind() { return genhist::HDW; }

  __device__ inline static
  void opAtom(volatile BETA* hist, volatile int* locks, int32_t idx, BETA v) {
    atomicAdd((uint32_t*) &hist[idx], (uint32_t)v);
  }
};

template<int RF>
struct SatAdd24 {
  typedef uint32_t BETA;
  typedef int32_t  ALPHA;

  __device__ __host__ inline static
  genhist::indval<BETA> f(const int32_t H, ALPHA pixel) {
    genhist::indval<BETA> res;
    const uint32_t ratio = max(1, H/RF);
    const uint32_t contraction = (((uint32_t)pixel) % ratio);
    res.index = contraction * RF;
    res.value = pixel % 4;
    return res;
  }

  __device__ __host__ inline static
  BETA ne() { return 0; }

  // 24-bits saturated addition
  __device__ __host__ inline static
  BETA opScal(BETA v1, BETA v2) {
    const uint32_t SAT_VAL24 = (1 << 24) - 1;
    uint32_t res;
    if(SAT_VAL24 - v1 < v2) {
      res = SAT_VAL24;
    } else {
      res = v1 + v2;
    }
    return res;
  }

  __device__ __host__ inline static
  genhist::AtomicPrim atomicKind() { return genhist::CAS; }

  __device__ inline static
  void opAtom(volatile BETA* hist, volatile int* locks, int32_t idx, BETA v) {
    genhist::atomCAS32bit<SatAdd24>(hist, locks, idx, v);
  }
};

template<int RF>
struct ArgMaxI64 {
  typedef uint64_t  BETA;
  typedef int32_t   ALPHA;

  __device__ __host__ inline static
  BETA pack64(uint32_t ind, uint32_t val) {
    uint64_t res = ind;
    uint64_t tmp = val;
    tmp = tmp << 32;
    res = res | tmp;
    return res;
  }

  __device__ __host__ inline static
  genhist::indval<uint32_t> unpack64(uint64_t t) {
    const uint64_t MASK32bits = 4294967295;
    genhist::indval<uint32_t> res;
    res.index = (uint32_t) (t & MASK32bits);
    res.value = (uint32_t) (t >> 32);
    return res;
  }

  __device__ __host__ inline static
  genhist::indval<BETA> f(const int32_t H, ALPHA pixel) {
    genhist::indval<BETA> res;

    const uint32_t ratio = max(1, H/RF);
    const uint32_t contraction = (((uint32_t)pixel) % ratio);
    res.index = contraction * RF;

    res.value = pack64( (uint32_t)pixel/64, (uint32_t)pixel );
    return res;
  }

  __device__ __host__ inline static
  BETA ne() { return 0; }



  __device__ __host__ inline static
  BETA opScal(BETA v1, BETA v2) {
    genhist::indval<uint32_t> arg1 = unpack64(v1);
    genhist::indval<uint32_t> arg2 = unpack64(v2);
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

  __device__ __host__ inline static
  genhist::AtomicPrim atomicKind() { return genhist::XCG; }

  __device__ inline static
  void opAtom(volatile BETA* hist, volatile int* locks, int32_t idx, BETA v) {
    genhist::atomXCG<ArgMaxI64>(hist, locks, idx, v);
  }
};

// Testing

template<class T>
void goldSeqHisto(const int32_t N, const int32_t H, typename T::ALPHA* input, typename T::BETA* histo) {
  typedef typename T::BETA BETA;
  zeroOut<T>(histo, H);
  for(int32_t i=0; i<N; i++) {
    struct genhist::indval<BETA> iv = T::f(H, input[i]);
    histo[iv.index] = T::opScal(histo[iv.index], iv.value);
  }
}

template<class T>
unsigned long
timeGoldSeqHisto(const int32_t N, const int32_t H, typename T::ALPHA* input, typename T::BETA* histo) {
  unsigned long int elapsed;
  struct timeval t_start, t_end, t_diff;
  gettimeofday(&t_start, NULL);

  for(int32_t q=0; q<CPU_RUNS; q++) {
    goldSeqHisto<T>(N, H, input, histo);
  }

  gettimeofday(&t_end, NULL);
  timeval_subtract(&t_diff, &t_end, &t_start);
  elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
  return (elapsed / CPU_RUNS);
}

template<class HP>
bool validate(typename HP::BETA* A, typename HP::BETA* B, unsigned int sizeAB) {
  const double EPS = 0.0000001;
  for(unsigned int i = 0; i < sizeAB; i++) {
    double diff = fabs( A[i] - B[i] );
    if ( diff > EPS ) {
      std::cout << "INVALID RESULT, index: " << i << " val_A: " << A[i] << ", val_B: " << B[i] << std::endl;;
      return false;
    }
  }
  return true;
}

template<class HP>
unsigned long
shmemHistoRunValid(const int32_t num_gpu_runs,
                   const int32_t H, const int32_t N,
                   typename HP::ALPHA* d_input,
                   typename HP::BETA* h_ref_histo) {
  typedef typename HP::BETA BETA;

  genhist::LocalMemoryGenHist<HP> do_genhist(genhist::rtx2080, H, N);

  // dry run
  do_genhist.exec(d_input);
  cudaDeviceSynchronize();
  gpuAssert( cudaPeekAtLastError() );

  unsigned long int elapsed;
  struct timeval t_start, t_end, t_diff;
  gettimeofday(&t_start, NULL);

  // measure runtime
  for(int32_t q=0; q<num_gpu_runs; q++) {
    do_genhist.exec(d_input);
  }
  cudaDeviceSynchronize();

  gettimeofday(&t_end, NULL);
  timeval_subtract(&t_diff, &t_end, &t_start);
  elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
  gpuAssert( cudaPeekAtLastError() );

  { // validate and free memory
    const size_t mem_size_histo  = H * sizeof(BETA);
    BETA* h_histo = (BETA*)malloc(mem_size_histo);
    cudaMemcpy(h_histo, do_genhist.result(), mem_size_histo, cudaMemcpyDeviceToHost);
    bool is_valid = validate<HP>(h_histo, h_ref_histo, H);

    free(h_histo);

    if(!is_valid) {
      printf("shmemHistoRunValid: Validation FAILS!\n");
      exit(3);
    }
  }

  return (elapsed/num_gpu_runs);
}

template< class HP >
uint64_t
glbmemHistoRunValid (const int32_t num_gpu_runs,
                     const int32_t B, const int32_t RF,
                     const int32_t H, const int32_t N,
                     typename HP::ALPHA* d_input,
                     typename HP::BETA* h_ref_histo) {
  typedef typename HP::BETA BETA;
  genhist::GlobalMemoryGenHist<HP> do_genhist(genhist::rtx2080, B, RF, H, N);

  // dry run
  do_genhist.exec(d_input);
  cudaDeviceSynchronize();
  gpuAssert( cudaPeekAtLastError() );

  uint64_t elapsed;
  struct timeval t_start, t_end, t_diff;
  gettimeofday(&t_start, NULL);

  // measure runtime
  for(int q=0; q<num_gpu_runs; q++) {
    do_genhist.exec(d_input);
  }
  cudaDeviceSynchronize();

  gettimeofday(&t_end, NULL);
  timeval_subtract(&t_diff, &t_end, &t_start);
  elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
  gpuAssert( cudaPeekAtLastError() );

  { // validate and free memory
    const size_t mem_size_histo  = H * sizeof(BETA);
    BETA* h_histo = (BETA*)malloc(mem_size_histo);
    cudaMemcpy(h_histo, do_genhist.result(), mem_size_histo, cudaMemcpyDeviceToHost);

    bool is_valid = validate<HP>(h_histo, h_ref_histo, H);

    free(h_histo);

    if(!is_valid) {
      printf("glbmemHistoRunValid: Validation FAILS!\n");
      exit(6);
    }
  }

  return (elapsed/num_gpu_runs);
}

template<int RF>
void runLocalMemDataset(int32_t* h_input, uint32_t* h_histo, int32_t* d_input, const int32_t N) {
  const int num_histos = 8;
  const int num_m_degs = 1;
  const int histo_sizes[num_histos] = {31, 127, 505, 2041, 6141, 12281, 24569, 49145};
  const int ks[num_m_degs] = { 33 };
  unsigned long runtimes[3][num_histos][num_m_degs];

  for(int i=0; i<num_histos; i++) {
    const int H = histo_sizes[i];

    { // FOR HDW
      goldSeqHisto< AddI32<RF> >(N, H, h_input, (int32_t*)h_histo);
      runtimes[0][i][0] = shmemHistoRunValid< AddI32<RF> >( GPU_RUNS, H, N, d_input, (int32_t*)h_histo);
    }

    { // FOR CAS
      goldSeqHisto< SatAdd24<RF> >(N, H, h_input, h_histo);
      runtimes[1][i][0] = shmemHistoRunValid< SatAdd24<RF> >( GPU_RUNS, H, N, d_input, h_histo);
    }

    { // FOR XCG
      goldSeqHisto< ArgMaxI64<RF> >(N, H, h_input, (uint64_t*)h_histo);
      runtimes[2][i][0] = shmemHistoRunValid< ArgMaxI64<RF> >( GPU_RUNS, H, N, d_input, (uint64_t*)h_histo);
    }
  }

  printTextTab<num_histos,num_m_degs>(runtimes, histo_sizes, ks, RF);
}

template<int RF>
void runGlobalMemDataset(int* h_input, uint32_t* h_histo, int* d_input, const int32_t N) {
  const int B = 256;
  const int num_histos = 7;
  const int num_m_degs = 1;
  const int histo_sizes[num_histos] = { 12281,  24569,  49145, 196607, 393215, 786431, 1572863 };
  const int subhisto_degs[num_m_degs] = { 33 };
  unsigned long runtimes[3][num_histos][num_m_degs];

  for(int i=0; i<num_histos; i++) {
    const int H = histo_sizes[i];

    { // For HDW
      goldSeqHisto< AddI32<RF> >(N, H, h_input, (int32_t*)h_histo);
      runtimes[0][i][0] = glbmemHistoRunValid< AddI32<RF> >( GPU_RUNS, B, RF, H, N, d_input, (int32_t*)h_histo);
    }

    { // FOR CAS
      goldSeqHisto< SatAdd24<RF> >(N, H, h_input, h_histo);
      runtimes[1][i][0] = glbmemHistoRunValid< SatAdd24<RF> >( GPU_RUNS, B, RF, H, N, d_input, h_histo);
    }

    { // FOR XCG
      goldSeqHisto< ArgMaxI64<RF> >(N, H, h_input, (uint64_t*)h_histo);
      runtimes[2][i][0] = glbmemHistoRunValid< ArgMaxI64<RF> >( GPU_RUNS, B, RF, H, N, d_input, (uint64_t*)h_histo);
    }
  }

  printTextTab<num_histos,num_m_degs>(runtimes, histo_sizes, subhisto_degs, RF);
}

void usage(const char *prog) {
  fprintf(stderr, "Usage: %s <local|global>\n", prog);
  exit(1);
}

int main(int argc, char **argv) {
  if (argc != 2 && argc != 8) {
    usage(argv[0]);
  }

  int run_local;
  if (strcmp(argv[1], "local") == 0) {
    run_local = 1;
  } else if (strcmp(argv[1], "global") == 0) {
    run_local = 0;
  } else {
    usage(argv[0]);
  }

  // set seed for rand()
  srand(2006);

  // 1. allocate host memory for input and histogram
  const unsigned int mem_size_input = sizeof(int) * INP_LEN;
  int* h_input = (int*) malloc(mem_size_input);
  const unsigned int mem_size_histo = sizeof(int) * Hmax;
  uint32_t* h_histo = (uint32_t*) malloc(mem_size_histo);

  // 2. initialize host memory
  randomInit(h_input, INP_LEN);
  zeroOut<SatAdd24<1> >(h_histo, Hmax);

  // 3. allocate device memory for input and copy from host
  int* d_input;
  cudaMalloc((void**) &d_input, mem_size_input);
  cudaMemcpy(d_input, h_input, mem_size_input, cudaMemcpyHostToDevice);

  if (run_local) {
    runLocalMemDataset<1> (h_input, h_histo, d_input, INP_LEN);
    runLocalMemDataset<63>(h_input, h_histo, d_input, INP_LEN);
  } else {
    runGlobalMemDataset<1> (h_input, h_histo, d_input, INP_LEN);
    runGlobalMemDataset<63>(h_input, h_histo, d_input, INP_LEN);

  }

  // 7. clean up memory
  free(h_input);
  free(h_histo);
  cudaFree(d_input);
}
