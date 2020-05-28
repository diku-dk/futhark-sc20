#include <cuda_runtime.h>
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
 
#include "histo-wrap.cu.h"

/***********************/
/*** Pretty printing ***/
/***********************/

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

template<int num_histos, int num_m_degs>
void printTextTab( const unsigned long runtimes[3][num_histos][num_m_degs]
                 , const int histo_sizes[num_histos]
                 , const int kms[num_m_degs]
                 , const int RF) {
    for(int k=0; k<3; k++) {
        printf("\n\n");

        printf(BOLD "%s, RF=%d\n" RESET,
               k == 0 ? "HWD" :
               k == 1 ? "CAS" :
               "XCG",
               RF);

        for(int i = 0; i<num_histos; i++) {
            printf(BOLD "\tH=%d" RESET, histo_sizes[i]);
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

template<int num_histos, int num_m_degs>
void printLaTex( const unsigned long runtimes[3][num_histos][num_m_degs]
               , const int histo_sizes[num_histos]
               , const int kms[num_m_degs]
               , const int R) {
    for(int k=0; k<3; k++) {
        printf("\\begin{tabular}{|l|l|l|l|l|l|l|l|}\\hline\n");
        if     (k==0) printf("ADD, R=%d", R);
        else if(k==1) printf("CAS, R=%d", R);
        else if(k==2) printf("XCG, R=%d", R);

        for(int i = 0; i<num_histos; i++) { printf("\t& H=%d", histo_sizes[i]); }
        printf("\\\\\\hline\n");
        for(int j=0; j<num_m_degs; j++) {
            if      (j==0)             printf("M=1 ");
            else if (j < num_m_degs-1) printf("M=%d", kms[j]);
            else                       printf("Ours");

            for(int i = 0; i<num_histos; i++) {
                printf("\t& %3.2f", runtimes[k][i][j]/1000.0);
            }
            printf("\\\\");
            if(j == (num_m_degs-1)) printf("\\hline");
            printf("\n");
        }
        printf("\\end{tabular}\n");
    }
}

template<int num_histos, int num_m_degs>
void printCSV(const char *csv, int k,
              const unsigned long runtimes[3][num_histos][num_m_degs],
              const int histo_sizes[num_histos],
              const int kms[num_m_degs],
              const char *mstr) {

    FILE* f = fopen(csv, "w");

    if (f == NULL) {
        fprintf(stderr, "Failed to open %s: %s\n", csv, strerror(errno));
        return;
    }

    fprintf(f, "M,");
    for(int i = 0; i<num_histos; i++) {
        fprintf(f, "%d", histo_sizes[i]);
        if (i != num_histos-1) {
            fprintf(f, ",");
        } else {
            fprintf(f, "\n");
        }
    }

    for(int j=0; j<num_m_degs; j++) {
        if (j < num_m_degs-1) {
            fprintf(f, "%s%d", mstr, kms[j]);
        } else {
            fprintf(f, "Auto");
        }
        for(int i = 0; i<num_histos; i++) {
            fprintf(f, ",%lu", runtimes[k][i][j]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

/****************************/
/*** HistogramDescriptors ***/
/****************************/

template<int RF>
struct AddI32 {
    typedef int32_t  BETA;
    typedef int32_t  ALPHA;

    __device__ __host__ inline static
    indval<BETA> f(const int32_t H, ALPHA pixel) {
        indval<BETA> res;
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
    AtomicPrim atomicKind() { return HWD; }

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
    indval<BETA> f(const int32_t H, ALPHA pixel) {
        indval<BETA> res;
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
    AtomicPrim atomicKind() { return CAS; }

    __device__ inline static
    void opAtom(volatile BETA* hist, volatile int* locks, int32_t idx, BETA v) {
        atomCAS32bit<SatAdd24>(hist, locks, idx, v);
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
    indval<uint32_t> unpack64(uint64_t t) {
        const uint64_t MASK32bits = 4294967295;
        indval<uint32_t> res;
        res.index = (uint32_t) (t & MASK32bits);
        res.value = (uint32_t) (t >> 32);
        return res;
    }

    __device__ __host__ inline static
    indval<BETA> f(const int32_t H, ALPHA pixel) {
        indval<BETA> res;

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
        indval<uint32_t> arg1 = unpack64(v1);
        indval<uint32_t> arg2 = unpack64(v2);
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
    AtomicPrim atomicKind() { return XCG; }

    __device__ inline static
    void opAtom(volatile BETA* hist, volatile int* locks, int32_t idx, BETA v) {
        atomXCG<ArgMaxI64>(hist, locks, idx, v);
    }
};

/****************/
/*** The meat ***/
/****************/


template<int RF>
void runLocalMemDataset(int32_t* h_input, uint32_t* h_histo, int32_t* d_input, const int32_t N,
                        const char *hwd_csv, const char *cas_csv, const char *xcg_csv) {
    const int num_histos = 8;
    const int num_m_degs = 1;
    const int histo_sizes[num_histos] = {31, 127, 505, 2041, 6141, 12281, 24569, 49145};
                                        //{/*25, 121, 505, 1024-7,*/ 2048-7, 4089, 6143, 12287, 24575, 49151};
                                        //{ 25, 57, 121, 249, 505, 1024-7, 4096-7, 12288-1, 24575, 4*12*1024-1 };
                                        //{ 64, 128, 256, 512 };
    const int ks[num_m_degs] = { 33 };
    unsigned long runtimes[3][num_histos][num_m_degs];

    for(int i=0; i<num_histos; i++) {
        const int H = histo_sizes[i];

        { // FOR HWD
            goldSeqHisto< AddI32<RF> >(N, H, h_input, (int32_t*)h_histo);
            runtimes[0][i][0] = shmemHistoRunValid< AddI32<RF> >( GPU_RUNS, H, N, d_input, (int32_t*)h_histo);
        }

        { // FOR CAS
            goldSeqHisto< SatAdd24<RF> >(N, H, h_input, h_histo);
            runtimes[1][i][0] = shmemHistoRunValid< SatAdd24<RF> >( GPU_RUNS, H, N, d_input, h_histo);
        }

        { // FOR XHG
            goldSeqHisto< ArgMaxI64<RF> >(N, H, h_input, (uint64_t*)h_histo);
            runtimes[2][i][0] = shmemHistoRunValid< ArgMaxI64<RF> >( GPU_RUNS, H, N, d_input, (uint64_t*)h_histo);
        }
    }

    //printTextTab<num_histos,num_m_degs>(runtimes, histo_sizes, ks, RF);
    printLaTex<num_histos,num_m_degs>  (runtimes, histo_sizes, ks, RF);

    if (hwd_csv) {
        printCSV(hwd_csv, 0, runtimes, histo_sizes, ks, "_");
    }
    if (cas_csv) {
        printCSV(cas_csv, 1, runtimes, histo_sizes, ks, "_");
    }
    if (xcg_csv) {
        printCSV(xcg_csv, 2, runtimes, histo_sizes, ks, "_");
    }
}

template<int RF>
void runGlobalMemDataset(int* h_input, uint32_t* h_histo, int* d_input, const int32_t N,
                        const char *hwd_csv, const char *cas_csv, const char *xcg_csv) {
    const int B = 256;
    const int T = NUM_THREADS(N);
    const int num_histos = 7;
    const int num_m_degs = 1;
    const int histo_sizes[num_histos] = { 12281,  24569,  49145
                                        , 196607, 393215, 786431, 1572863 };
                                        //{ 1*12*1024-algn,  2*12*1024-algn,  4*12*1024-algn
                                        //, 8*12*1024-algn, 16*12*1024-algn, 32*12*1024-algn
                                        //, 64*12*1024-algn, 128*12*1024-algn };
    const int subhisto_degs[num_m_degs] = { 33 };    
    unsigned long runtimes[3][num_histos][num_m_degs];

    for(int i=0; i<num_histos; i++) {
        const int H = histo_sizes[i];

        { // For HWD
        	goldSeqHisto< AddI32<RF> >(N, H, h_input, (int32_t*)h_histo);
        	runtimes[0][i][0] = glbmemHistoRunValid< AddI32<RF> >( GPU_RUNS, B, RF, H, N, d_input, (int32_t*)h_histo);
        }

        { // FOR CAS
            goldSeqHisto< SatAdd24<RF> >(N, H, h_input, h_histo);
            runtimes[1][i][0] = glbmemHistoRunValid< SatAdd24<RF> >( GPU_RUNS, B, RF, H, N, d_input, h_histo);
        }

        { // FOR XHG
            goldSeqHisto< ArgMaxI64<RF> >(N, H, h_input, (uint64_t*)h_histo);
            runtimes[2][i][0] = glbmemHistoRunValid< ArgMaxI64<RF> >( GPU_RUNS, B, RF, H, N, d_input, (uint64_t*)h_histo);
        }
    }

    //printTextTab<num_histos,num_m_degs>(runtimes, histo_sizes, subhisto_degs, RF);
    printLaTex<num_histos,num_m_degs>(runtimes, histo_sizes, subhisto_degs, RF);

    if (hwd_csv) {
        printCSV(hwd_csv, 0, runtimes, histo_sizes, subhisto_degs, "=");
    }
    if (cas_csv) {
        printCSV(cas_csv, 1, runtimes, histo_sizes, subhisto_degs, "=");
    }
    if (xcg_csv) {
        printCSV(xcg_csv, 2, runtimes, histo_sizes, subhisto_degs, "=");
    }
}

void usage(const char *prog) {
    fprintf(stderr, "Usage: %s <local|global>\n", prog);
    exit(1);
}

/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
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

    const char *hwd_csv_1  = NULL;
    const char *cas_csv_1  = NULL;
    const char *xcg_csv_1  = NULL;
    const char *hwd_csv_63 = NULL;
    const char *cas_csv_63 = NULL;
    const char *xcg_csv_63 = NULL;

    if (argc == 8) {
        hwd_csv_1 = argv[2];
        cas_csv_1 = argv[3];
        xcg_csv_1 = argv[4];
        hwd_csv_63 = argv[5];
        cas_csv_63 = argv[6];
        xcg_csv_63 = argv[7];
    }

    // set seed for rand()
    srand(2006);

    // remember to initialize the gpu device properties!
    initGPUprops();

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
        runLocalMemDataset<1> (h_input, h_histo, d_input, INP_LEN,
                           	   hwd_csv_1, cas_csv_1, xcg_csv_1);
        runLocalMemDataset<63>(h_input, h_histo, d_input, INP_LEN,
                           	   hwd_csv_63, cas_csv_63, xcg_csv_63);
    } else {
        runGlobalMemDataset<1> (h_input, h_histo, d_input, INP_LEN,
                            	hwd_csv_1, cas_csv_1, xcg_csv_1);
        runGlobalMemDataset<63>(h_input, h_histo, d_input, INP_LEN,
                            	hwd_csv_63, cas_csv_63, xcg_csv_63);

    }

    // 7. clean up memory
    free(h_input);
    free(h_histo);
    cudaFree(d_input);
}
