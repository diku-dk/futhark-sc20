#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h> 

#define BLOCK       1024
#define GPU_RUNS    200
#define CPU_RUNS    1

#define INP_LEN     50000000
#define Hmax        100000
 
#ifndef DEBUG_INFO
#define DEBUG_INFO  0
#endif

#ifndef RACE_FACT
#define RACE_FACT   8
#endif

#ifndef LOCMEMW_PERTHD
#define LOCMEMW_PERTHD 12
#endif



cudaDeviceProp prop;
unsigned int HWD;
unsigned int SH_MEM_SZ;
unsigned int BLOCK_SZ;

#define NUM_THREADS(n)  min(n, HWD)

#include "histo-kernels.cu.h"
#include "histo-wrap.cu.h"


int optimSubHistoDeg(const AtomicPrim prim_kind, const int Q, const int H) {
    const int el_size = (prim_kind == XCHG)? 2*sizeof(int) : sizeof(int);
    const int m = ((Q*4 / el_size) * BLOCK) / H;
    //const int coop = (BLOCK + m - 1) / m;
    //printf("COOP LEVEL: %d, subhistogram degree: %d\n", coop, m);
    return min(m, BLOCK);
}


void testLocMemAlignmentProblem(const int H, int* h_input, int* h_histo, int* d_input) {
        
        unsigned long tm_seq = goldSeqHisto(INP_LEN, H, h_input, h_histo);
        printf("Histogram Sequential        took: %lu microsecs\n", tm_seq);

        int histos_per_block = 3*BLOCK/min(H, BLOCK);

        unsigned long tm_cas = locMemHwdAddCoop(CAS, INP_LEN, H, histos_per_block, d_input, h_histo);
        printf("Histogram H=%d Local-Mem CAS with subhisto-degree %d took: %lu microsecs\n", H, histos_per_block, tm_cas);

        histos_per_block = 6*BLOCK/min(H, BLOCK);
        tm_cas = locMemHwdAddCoop(CAS, INP_LEN, H, histos_per_block, d_input, h_histo);
        printf("Histogram H=%d Local-Mem CAS with subhisto-degree %d took: %lu microsecs\n", H, histos_per_block, tm_cas);

        histos_per_block = optimSubHistoDeg(CAS, LOCMEMW_PERTHD, H); 
        tm_cas = locMemHwdAddCoop(CAS, INP_LEN, H, histos_per_block, d_input, h_histo);
        printf("Histogram H=%d Local-Mem CAS with subhisto-degree %d took: %lu microsecs\n", H, histos_per_block, tm_cas);
}



void runLocalMemDataset(int* h_input, int* h_histo, int* d_input) {
    const int num_histos = 5;
    const int num_m_degs = 5;
    const int histo_sizes[num_histos] = { 25, 57, 121, 249, 505 }; //{ 64, 128, 256, 512 };
    //const AtomicPrim atomic_kinds[3] = {ADD, CAS, XCHG};

    unsigned long runtimes[3][num_histos][num_m_degs];

    for(int i=0; i<num_histos; i++) {
        const int H = histo_sizes[i];
        const int m_opt = optimSubHistoDeg(ADD, LOCMEMW_PERTHD, H);

        const int min_HB = min(H,BLOCK);
        const int subhisto_degs[5] = { 1, BLOCK/min_HB, 3*BLOCK/min_HB, 6*BLOCK/min_HB, m_opt }; 
        //{ m_opt, (8*BLOCK) / min_HB, (4*BLOCK) / min_HB, (1*BLOCK) / min_HB, 1};

        goldSeqHisto(INP_LEN, H, h_input, h_histo);

        for(int j=0; j<num_m_degs; j++) {
            const int histos_per_block = subhisto_degs[j];
            runtimes[0][i][j] = locMemHwdAddCoop(ADD, INP_LEN, H, histos_per_block, d_input, h_histo);
            runtimes[1][i][j] = locMemHwdAddCoop(CAS, INP_LEN, H, histos_per_block, d_input, h_histo);
            runtimes[2][i][j] = locMemHwdAddCoop(XCHG, INP_LEN, H, max(histos_per_block/2,1), d_input, h_histo);
        }
    }

    printTextTab(runtimes, histo_sizes, RACE_FACT);
    //printLaTex  (runtimes, histo_sizes, RACE_FACT);
}


/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
int main() {
    // set seed for rand()
    srand(2006);

    { // 0. querry the hardware
        int nDevices;
        cudaGetDeviceCount(&nDevices);
  
        cudaGetDeviceProperties(&prop, 0);
        HWD = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
        BLOCK_SZ = prop.maxThreadsPerBlock;
        SH_MEM_SZ = prop.sharedMemPerBlock;
        if (DEBUG_INFO) {
            printf("Device name: %s\n", prop.name);
            printf("Number of hardware threads: %d\n", HWD);
            printf("Block size: %d\n", BLOCK_SZ);
            printf("Shared memory size: %d\n", SH_MEM_SZ);
            puts("====");
        }
    }

 
    // 1. allocate host memory for input and histogram
    const unsigned int mem_size_input = sizeof(int) * INP_LEN;
    int* h_input = (int*) malloc(mem_size_input);
    const unsigned int mem_size_histo = sizeof(int) * Hmax;
    int* h_histo = (int*) malloc(mem_size_histo);
 
    // 2. initialize host memory
    randomInit(h_input, INP_LEN);
    zeroOut(h_histo, Hmax);
    
    // 3. allocate device memory for input and copy from host
    int* d_input;
    cudaMalloc((void**) &d_input, mem_size_input);
    cudaMemcpy(d_input, h_input, mem_size_input, cudaMemcpyHostToDevice);
 
#if 0
    { // 5. compute a bunch of histograms
        const int H = 128;
        
        unsigned long tm_seq = goldSeqHisto(INP_LEN, H, h_input, h_histo);
        printf("Histogram Sequential        took: %lu microsecs\n", tm_seq);

        int histos_per_block = BLOCK/32;
        //int histos_per_block = optimSubHistoDeg(CAS, 12, H); 
        unsigned long tm_add = locMemHwdAddCoop(ADD, INP_LEN, H, histos_per_block, d_input, h_histo);
        printf("Histogram Local-Mem ADD with subhisto-degree %d took: %lu microsecs\n", histos_per_block, tm_add);

        unsigned long tm_cas = locMemHwdAddCoop(CAS, INP_LEN, H, histos_per_block, d_input, h_histo);
        printf("Histogram Local-Mem CAS with subhisto-degree %d took: %lu microsecs\n", histos_per_block, tm_cas);

        //coop = optimalCoop(XCHG, 12, H);
        unsigned long tm_xch = locMemHwdAddCoop(XCHG, INP_LEN, H, histos_per_block/2, d_input, h_histo);
        printf("Histogram Local-Mem XCG with subhisto-degree %d took: %lu microsecs\n", histos_per_block, tm_xch);
    }
#endif

#if 0
    { // 5. compute a bunch of histograms
        for(int i=0; i<34; i++)
            testLocMemAlignmentProblem(31+i, h_input, h_histo, d_input);
    }
#endif

#if 1
    runLocalMemDataset(h_input, h_histo, d_input);
#endif

    // 7. clean up memory
    free(h_input);
    free(h_histo);
    cudaFree(d_input);
}

