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
    #define MF 5632
    #define RF 0.75
#else // GPU_KIND==2
    #define MF 1024
    #define RF 0.5
#endif

#define GLB_K_MIN   2

#ifndef RACE_FACT
#define RACE_FACT   32 //32  // = H / (Num_Distinct_Pts)
#endif

#ifndef STRIDE
#define STRIDE      16  // = (Max_Ind_Pt - Min_Ind_Pt) / Num_Distinct_Pts
#endif

#define CLelmsz     16 // how many elements fit on a L2 cache line

#define L2Cache     (MF*1024)
#define L2Fract     0.4

#if 1
  #define CTGRACE     0
  #define RACE_EXPNS MAX(1.0, RF * (((float)RACE_FACT)/CLelmsz) * ( (CLelmsz>STRIDE) ? (CLelmsz/STRIDE) : 1 ) )
#else
  #define CTGRACE     1
  #define SHRINK_FACT (0.75*RACE_FACT) //0.625
  #if CTGRACE
    #define RACE_EXPNS  MAX(1.0, SHRINK_FACT)
  #else
    #define RACE_EXPNS  MAX(1.0, SHRINK_FACT / 16)
  #endif
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
    const int elms_per_block = (N + BLOCK - 1) / BLOCK;
    const int el_size = (prim_kind == XCHG)? 3*sizeof(int) : sizeof(int);

    float m_prime = MIN( (lmem*1.0 / el_size), (float)elms_per_block ) / H;

    if (prim_kind == ADD) {
        *M = max(1, min( (int)floor(m_prime), BLOCK ) );
    } else {
        float m = max(1.0, m_prime);
        const float RFC = MIN( (float)RACE_FACT, 32.0*pow(RACE_FACT/32.0, 0.33) );
        float f_prime = (BLOCK*RFC) / (m*m*H);
        float f_lower = (prim_kind==CAS) ? ceil(f_prime) : floor(f_prime);
        const int  f  = max(1, (int)f_lower);
        *M = max(1, min( (int)floor((prim_kind==CAS)? m*f : m_prime*f), BLOCK));

        printf("In computeLocM: prim-kind %d, H %d, result f: %f, m: %f, M: %d\n"
              , prim_kind, H, f_prime, m, *M);
    }
    const int len = lmem / (el_size * (*M));
    *num_chunks = (H + len - 1) / len;
}



void autoLocSubHistoDeg0(const AtomicPrim prim_kind, const int H, const int N, int* M, int* num_chunks) {
    const int lmem = LOCMEMW_PERTHD * BLOCK * 4;
    const int elms_per_block = (N + BLOCK - 1) / BLOCK;
    const int el_size_tot = (prim_kind == XCHG)? 3*sizeof(int) : sizeof(int);
    const int el_size = (prim_kind == XCHG)? 2*sizeof(int) : sizeof(int);

    float m     = MIN( (lmem*1.0 / el_size)    , (float)elms_per_block ) / H;
    float m_tot = MIN( (lmem*1.0 / el_size_tot), (float)elms_per_block ) / H;

    if (prim_kind == ADD) {
        *M = max(1, min( (int)floor(m), BLOCK ) );
    } else {
        m = max(1.0, m);
        const float c = BLOCK / m;
        const float RFC = MIN( (float)RACE_FACT, 32.0*pow(RACE_FACT/32.0, 0.33) );
        float tmp1 = c*RFC / (m * H);
        if (m_tot / m > tmp1) {
            *M = min( (int)MAX(floor(m_tot), 1.0), BLOCK );
        } else {
            float tmp = (prim_kind==CAS) ? ceil(tmp1) : floor(tmp1);
            float f = MAX( 1.0, tmp );
            *M = min( (int) floor(m*f), BLOCK);
        }
        printf("In computeLocM: prim-kind %d, H %d, result f: %f, m: %f, M: %d\n"
              , prim_kind, H, tmp1, m, *M);
    }
    const int len = lmem / (el_size_tot * (*M));
    *num_chunks = (H + len - 1) / len;


    // cooperation level can be define independently as
    //     C = min(H/k, B) for some smallish k, or
    // derived from M as
    //     C = ceil(BLOCK/M)
    //const int coop = (BLOCK + m - 1) / m;
    //printf("COOP LEVEL: %d, subhistogram degree: %d\n", coop, m);
    //return min(m, BLOCK);
}

int autoGlbSubHistoDeg(
                const AtomicPrim prim_kind, const int H, const int N, const int T, const int L2
) {
    const int el_size = (prim_kind == XCHG)? 3*sizeof(int) : sizeof(int);
    const float frac  = L2Fract * RACE_EXPNS;
    const float k_max = MIN( frac * (L2 / el_size) / T, ((float)N)/T );
    const float coop_h = (prim_kind == ADD) ? (2.0*H) / k_max : (1.0*H) / k_max; 
    const float coop  = MIN( (float)T, coop_h );
    return max(1, (int) (T / coop));
}

void autoGlbChunksSubhists0(
                const AtomicPrim prim_kind, const int H, const int N, const int T, const int L2,
                int* M, int* num_chunks ) {
    const int el_size = (prim_kind == XCHG)?
                        3*sizeof(int) : sizeof(int);
    
    const float  optim_k_min = GLB_K_MIN;
    const float  coop  = MIN( (float)T, H/optim_k_min );
    const int    Mdeg  = max(1, (int) (T / coop));
    const size_t totsz = Mdeg * H;
    const size_t L2csz = L2Fract * (L2 / el_size) * RACE_EXPNS;
    const int num_chks = (totsz + L2csz - 1) / L2csz;
    const int Hnew     = (H + num_chks - 1) / num_chks;

    *num_chunks = num_chks;
    *M = autoGlbSubHistoDeg(prim_kind, Hnew, N, T, L2);

    printf( "CHUNKING branch: optim_k_min: %f, coop: %f, Mdeg: %d, Hold: %d, Hnew: %d, num_chunks: %d, M: %d\n"
          , optim_k_min, coop, Mdeg, H, Hnew, *num_chunks, *M );
}

void autoGlbChunksSubhists(
                const AtomicPrim prim_kind, const int H, const int N, const int T, const int L2,
                int* M, int* num_chunks ) {
    const int   el_size = (prim_kind == XCHG)?
                          3*sizeof(int) : sizeof(int);
    const float optim_k_min = GLB_K_MIN;
        
    // first part
    float race_exp = max(1.0, (1.0 * RF * RACE_FACT) / (CLelmsz / el_size) );
    float coop_min = MIN( (float)T, H/optim_k_min );
    const int Mdeg  = max(1, (int) (T / coop_min));
    const int H_chk = ( L2Fract * ((1.0*L2Cache) / el_size) * race_exp ) / Mdeg;
    *num_chunks = (H + H_chk - 1) / H_chk;

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

    //printTextTab<num_histos,num_m_degs>(runtimes, histo_sizes, ks, RACE_FACT);
    printLaTex<num_histos,num_m_degs>  (runtimes, histo_sizes, ks, RACE_FACT);
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

    printf("Running Histo in Global Mem: RACE_FACT: %d, STRIDE: %d, RACE_EXPNS: %f, L2Cache:%d, L2Fract: %f\n",
           RACE_FACT, STRIDE, RACE_EXPNS, L2Cache, L2Fract);

    //printTextTab<num_histos,num_m_degs>(runtimes, histo_sizes, subhisto_degs, RACE_FACT);
    printLaTex<num_histos,num_m_degs>(runtimes, histo_sizes, subhisto_degs, RACE_FACT);
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
