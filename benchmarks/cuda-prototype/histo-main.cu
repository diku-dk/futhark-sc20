#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h> 

#define MIN(a,b)    (((a) < (b)) ? (a) : (b))
#define MAX(a,b)    (((a) < (b)) ? (b) : (a)) 

#define GPU_KIND    1 // 1 -> RTX2080Ti; 2 -> GTX1050Ti

#if (GPU_KIND==1)
    #define MF 5632   // 4096 for RTX2070
    #define K_RF 0.75
#else // GPU_KIND==2
    #define MF 1024
    #define K_RF 0.5
#endif

#define GLB_K_MIN   2

#ifndef RF
#define RF   64 //32  // = H / (Num_Distinct_Pts)
#endif

#ifndef STRIDE
#define STRIDE      64  // = (Max_Ind_Pt - Min_Ind_Pt) / Num_Distinct_Pts
#endif

#define CLelmsz     16 // how many elements fit on a L2 cache line

#define L2Cache     (MF*1024)
#define L2Fract     0.4

#if 1
  #define CTGRACE     0
#else
  #define CTGRACE     1
  #define SHRINK_FACT (0.75*RF) //0.625
#endif

#define BLOCK       1024
#define GPU_RUNS    50
#define CPU_RUNS    1

#define INP_LEN     50000000
#define Hmax        4000000
 
#ifndef DEBUG_INFO
#define DEBUG_INFO  1
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

void autoLocSubHistoDeg(const AtomicPrim prim_kind, const int H, const int N, int* M, int* num_chunks) {
    const int lmem = LOCMEMW_PERTHD * BLOCK * 4;
    const int num_blocks = (NUM_THREADS(N) + BLOCK - 1) / BLOCK;
    const int q_small = 2;
    const int work_asymp_M_max = N / (q_small*num_blocks*H);

    const int elms_per_block = (N + num_blocks - 1) / num_blocks; //(N + BLOCK - 1) / BLOCK;
    const int el_size = (prim_kind == XCHG)? 3*sizeof(int) : sizeof(int);
    float m_prime = MIN( (lmem*1.0 / el_size), (float)elms_per_block ) / H;

    if (prim_kind == ADD) {
        *M = max(1, min( (int)floor(m_prime), BLOCK ) );
        *M = min(*M, work_asymp_M_max);
    } else {
        float m = max(1.0, m_prime);
        const float RFC = MIN( (float)RF, 32.0*pow(RF/32.0, 0.33) );
        float f_prime = (BLOCK*RFC) / (m*m*H);
        float f_lower = (prim_kind==CAS) ? ceil(f_prime) : floor(f_prime);
        const int  f  = max(1, (int)f_lower);
        *M = max(1, min( (int)floor((prim_kind==CAS)? m*f : m_prime*f), BLOCK));
        *M = min(*M, work_asymp_M_max);

        printf("In computeLocM: prim-kind %d, H %d, result f: %f, m: %f, M: %d\n"
              , prim_kind, H, f_prime, m, *M);
    }
    const int len = lmem / (el_size * (*M));
    *num_chunks = (H + len - 1) / len;
}

void autoGlbChunksSubhists(
                const AtomicPrim prim_kind, const int H, const int N, const int T, const int L2,
                int* M, int* num_chunks ) {
    // For the computation of avg_size on XCHG:
    //   In principle we average the size of the lock and of the element-type of histogram
    //   But Futhark uses a tuple-of-array rep: hence we need to average the lock together
    //     with each element type from the tuple.
    const int   avg_size= (prim_kind == XCHG)? 3*sizeof(int)/2 : sizeof(int);
    const int   el_size = (prim_kind == XCHG)? 3*sizeof(int) : sizeof(int);
    const float optim_k_min = GLB_K_MIN;
    const int q_small = 2;
    const int work_asymp_M_max = N / (q_small*H);
        
    // first part
    float race_exp = max(1.0, (1.0 * K_RF * RF) / ( (4.0*CLelmsz) / avg_size) );
    float coop_min = MIN( (float)T, H/optim_k_min );
    const int Mdeg  = min(work_asymp_M_max, max(1, (int) (T / coop_min)));
    *num_chunks = (int)ceil( Mdeg*H / ( L2Fract * ((1.0*L2Cache) / el_size) * race_exp ) );
    const int H_chk = (int)ceil( H / (*num_chunks) );
    //const int H_chk = ( L2Fract * ((1.0*L2Cache) / el_size) * race_exp ) / Mdeg;
    //*num_chunks = (H + H_chk - 1) / H_chk;

    // second part
    const float u = (prim_kind == ADD) ? 2.0 : 1.0;
    const float k_max= MIN( L2Fract * ( (1.0*L2Cache) / el_size ) * race_exp, (float)N ) / T;
    const float coop = MIN( T, (u * H_chk) / k_max );
    *M = max( 1, (int)floor(T/coop) );
     
    printf( "CHUNKING branch: optim_k_min: %f, coop: %f, Mdeg: %d, Hold: %d, Hnew: %d, num_chunks: %d, M: %d\n"
          , optim_k_min, coop_min, Mdeg, H, H_chk, *num_chunks, *M );
}


void runLocalMemDataset(int* h_input, uint32_t* h_histo, int* d_input) {
    const int num_histos = 8;
    const int num_m_degs = 6;
    const int histo_sizes[num_histos] = {25, 121, 505, 2041, 6143, 12287, 24575, 49151};
                                        //{/*25, 121, 505, 1024-7,*/ 2048-7, 4089, 6143, 12287, 24575, 49151};
                                        //{ 25, 57, 121, 249, 505, 1024-7, 4096-7, 12288-1, 24575, 4*12*1024-1 };
                                        //{ 64, 128, 256, 512 };
    //const AtomicPrim atomic_kinds[3] = {ADD, CAS, XCHG};
    const int ks[num_m_degs] = { 0, 1, 3, 6, 9, 33 };
    unsigned long runtimes[3][num_histos][num_m_degs];

    for(int i=0; i<num_histos; i++) {
        const int H = histo_sizes[i];
        int m_opt, num_chunks;
        autoLocSubHistoDeg(ADD, H, INP_LEN, &m_opt, &num_chunks);

        // COSMIN is here: this is tricky to adapt since it stores only the
        //                 subhistos and not the num_chunks factor.
        const int min_HB = min(H,BLOCK);
        const int subhisto_degs[num_m_degs] = { 1, BLOCK/min_HB, 3*BLOCK/min_HB, 6*BLOCK/min_HB, 9*BLOCK/min_HB, m_opt };

        { // FOR ADD
            goldSeqHisto<ADD>(INP_LEN, H, h_input, h_histo);

            for(int j=0; j<num_m_degs; j++) {
              if(j == num_m_degs-1) {
                int histos_per_block, num_chunks;
                autoLocSubHistoDeg(ADD,  H, INP_LEN, &histos_per_block, &num_chunks);
                runtimes[0][i][j] = locMemHwdAddCoop(ADD,  INP_LEN, H, histos_per_block, num_chunks, d_input, h_histo);
              } else {
                const int lmem = LOCMEMW_PERTHD*BLOCK, M = subhisto_degs[j];
                int len = lmem / M, num_chunks = (H + len - 1) / len;
                runtimes[0][i][j] = locMemHwdAddCoop(ADD,  INP_LEN, H, M, num_chunks, d_input, h_histo);
              }
            }
        }

        { // FOR CAS
            goldSeqHisto<CAS>(INP_LEN, H, h_input, h_histo);
            for(int j=0; j<num_m_degs; j++) {
              if(j == num_m_degs-1) {
                int histos_per_block, num_chunks;
                autoLocSubHistoDeg(CAS,  H, INP_LEN, &histos_per_block, &num_chunks);
                runtimes[1][i][j] = locMemHwdAddCoop(CAS,  INP_LEN, H, histos_per_block, num_chunks, d_input, h_histo);
              } else {
                const int lmem = LOCMEMW_PERTHD*BLOCK, M = subhisto_degs[j];
                int len = lmem / M, num_chunks = (H + len - 1) / len;
                runtimes[1][i][j] = locMemHwdAddCoop(CAS,  INP_LEN, H, M, num_chunks, d_input, h_histo);
              }
            }
        }

        { // FOR XHCG
            goldSeqHisto<XCHG>(INP_LEN, H, h_input, h_histo);

            for(int j=0; j<num_m_degs; j++) {
              if(j == num_m_degs-1) {
                int histos_per_block, num_chunks;
                autoLocSubHistoDeg(XCHG, H, INP_LEN, &histos_per_block, &num_chunks);
                runtimes[2][i][j] = locMemHwdAddCoop(XCHG, INP_LEN, H, histos_per_block, num_chunks, d_input, h_histo); 
              } else {
                const int lmem = LOCMEMW_PERTHD*BLOCK, M = subhisto_degs[j];
                int len = lmem / (3*M), num_chunks = (H + len - 1) / len;
                runtimes[2][i][j] = locMemHwdAddCoop(XCHG, INP_LEN, H, M, num_chunks, d_input, h_histo);
              }
            }
        }

    }

    //printTextTab<num_histos,num_m_degs>(runtimes, histo_sizes, ks, RF);
    printLaTex<num_histos,num_m_degs>  (runtimes, histo_sizes, ks, RF);
}


void runGlobalMemDataset(int* h_input, uint32_t* h_histo, int* d_input) {
    const int B = 256;
    const int T = NUM_THREADS(INP_LEN);
    const int num_histos = 7;
    const int num_m_degs = 6;
    const int algn = 1;
    const int histo_sizes[num_histos] = { 1*12*1024-algn,  2*12*1024-algn,  4*12*1024-algn
                                        , 16*12*1024-algn, 32*12*1024-algn
                                        , 64*12*1024-algn, 128*12*1024-algn };
                                        //{ 1*12*1024-algn,  2*12*1024-algn,  4*12*1024-algn
                                        //, 8*12*1024-algn, 16*12*1024-algn, 32*12*1024-algn
                                        //, 64*12*1024-algn, 128*12*1024-algn };
    const int subhisto_degs[num_m_degs] = { 1, 4, 8, 16, 32, 33 };    
    unsigned long runtimes[3][num_histos][num_m_degs];

    for(int i=0; i<num_histos; i++) {
        const int H = histo_sizes[i];

        { // For ADD
            goldSeqHisto<ADD>(INP_LEN, H, h_input, h_histo);

            for(int j=0; j<num_m_degs; j++) {
                int M_add, num_chunks_add;
                if(j == num_m_degs-1) {
                    autoGlbChunksSubhists(ADD,  H, INP_LEN, T, L2Cache, &M_add, &num_chunks_add);
                } else {
                    num_chunks_add = 1; M_add = subhisto_degs[j];
                }
                if(j==(num_m_degs-1))
                    printf("Our M_add: %d, num_chunks_cas: %d, for H: %d\n", M_add, num_chunks_add, H);

                runtimes[0][i][j] = glbMemHwdAddCoop(ADD,  INP_LEN, H, B, M_add, num_chunks_add, d_input, h_histo);
            }
        }

        { // For CAS
            goldSeqHisto<CAS>(INP_LEN, H, h_input, h_histo);

            for(int j=0; j<num_m_degs; j++) {
                int M_cas, num_chunks_cas;
                if(j == num_m_degs-1) {
                    autoGlbChunksSubhists(CAS,  H, INP_LEN, T, L2Cache, &M_cas, &num_chunks_cas);
                } else {
                    num_chunks_cas = 1; M_cas = subhisto_degs[j];
                }
                if(j==(num_m_degs-1))
                    printf("Our M_cas: %d, num_chunks_cas: %d, for H: %d\n", M_cas, num_chunks_cas, H);

                runtimes[1][i][j] = glbMemHwdAddCoop(CAS,  INP_LEN, H, B, M_cas, num_chunks_cas, d_input, h_histo);
            }
        }

        { // For XCHG
            goldSeqHisto<XCHG>(INP_LEN, H, h_input, h_histo);

            for(int j=0; j<num_m_degs; j++) {
                int M_lck, num_chunks_lck;
                if(j == num_m_degs-1) {
                    autoGlbChunksSubhists(XCHG, H, INP_LEN, T, L2Cache, &M_lck, &num_chunks_lck);
                } else {
                    num_chunks_lck = 1; M_lck = (subhisto_degs[j]+2)/3;
                }
                if(j==(num_m_degs-1))
                    printf("Our M_lck: %d, num_chunks_lck: %d, for H: %d\n", M_lck, num_chunks_lck, H);

                runtimes[2][i][j] = glbMemHwdAddCoop(XCHG, INP_LEN, H, B, M_lck, num_chunks_lck, d_input, h_histo);
            }
        }
    }

    printf("Running Histo in Global Mem: RACE_FACT: %d, STRIDE: %d, L2Cache:%d, L2Fract: %f\n",
           RF, STRIDE, L2Cache, L2Fract);

    //printTextTab<num_histos,num_m_degs>(runtimes, histo_sizes, subhisto_degs, RF);
    printLaTex<num_histos,num_m_degs>(runtimes, histo_sizes, subhisto_degs, RF);
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
    uint32_t* h_histo = (uint32_t*) malloc(mem_size_histo);
 
    // 2. initialize host memory
    randomInit(h_input, INP_LEN);
    zeroOut<uint32_t>(h_histo, Hmax);
    
    // 3. allocate device memory for input and copy from host
    int* d_input;
    cudaMalloc((void**) &d_input, mem_size_input);
    cudaMemcpy(d_input, h_input, mem_size_input, cudaMemcpyHostToDevice);
 
#if 0
    runLocalMemDataset(h_input, h_histo, d_input);
#endif

#if 1
    runGlobalMemDataset(h_input, h_histo, d_input);
#endif
    // 7. clean up memory
    free(h_input);
    free(h_histo);
    cudaFree(d_input);
}
