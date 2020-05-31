#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <errno.h>
#include <string.h>

#define MIN(a,b)    (((a) < (b)) ? (a) : (b))
#define MAX(a,b)    (((a) < (b)) ? (b) : (a))
 
#ifndef DEBUG_INFO
#define DEBUG_INFO  0
#endif

#ifndef LOCMEMW_PERTHD
#define LOCMEMW_PERTHD 12
#endif

#define CPU_RUNS 10
#define GPU_RUNS 5000


unsigned int HWD;
unsigned int SH_MEM_SZ;
unsigned int MAX_BLOCK_SZ;

#define lgWARP      5
#define WARP        (1<<lgWARP)

//#define NUM_BLOCKS_SCAN     1024
#define ELEMS_PER_THREAD    11//12

#define NUM_THREADS(n)  min(n, HWD)

typedef float real;
const int  ntype = 19;
const real facel = 0.5;
const real nul     =  0.0;
const real one     =  1.0;
const real six     =  6.0;
const real twelve  = 12.0;

#include "util.h"
#include "gromacs-kernels.cu.h"
#include "gromacs-wrap.cu.h"

/***********************/
/*** Pretty printing ***/
/***********************/


void usage(const char *prog) {
    fprintf(stderr, "Usage: %s <input-file> <nri_max> <nrj_max> <num_particles_max> [output_file]\n", prog);
    exit(1);
}


/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////

// nvcc -O4 -arch=compute_35 -prec-sqrt=true gromacs-main.cu
// ./a.out data/huge-indarr.txt.in 36000 1600000 32000
int main(int argc, char **argv) {
    if (argc != 5 && argc != 6) {
        usage(argv[0]);
    }

    // 1. reading/constructing the dataset

    int *jindex = NULL, *iinr = NULL, *jjnr = NULL, *shift = NULL, *types = NULL;
    unsigned int nri = 0, nrj = 0, num_particles = 0; 

    unsigned int nri_max = atoi(argv[2]),
                 nrj_max = atoi(argv[3]),
                 prt_max = atoi(argv[4]);

    { // read indirect arrays
        char* file_name = argv[1];
        int jindex_len, iinr_len, jjnr_len, shift_len, types_len;
        unsigned int iinr_maxlen   = nri_max; //36000;
        unsigned int jindex_maxlen = nri_max + 16;
        unsigned int jjnr_maxlen   = nrj_max; //1600000;
        unsigned int types_maxlen  = prt_max; //32000;
        jindex = (int*) malloc(jindex_maxlen * sizeof(int));
        iinr   = (int*) malloc(iinr_maxlen   * sizeof(int));
        jjnr   = (int*) malloc(jjnr_maxlen   * sizeof(int));
        shift  = (int*) malloc(iinr_maxlen   * sizeof(int));
        types  = (int*) malloc(types_maxlen  * sizeof(int));
        readDataset ( file_name
                    , jindex, jindex_maxlen, &jindex_len
                    , iinr,   iinr_maxlen,   &iinr_len
                    , jjnr,   jjnr_maxlen,   &jjnr_len
                    , shift,  iinr_maxlen,   &shift_len
                    , types,  types_maxlen,  &types_len
                    );

        if ( (jindex_len >= jindex_maxlen) ||
             (iinr_len   >= iinr_maxlen  ) ||
             (jjnr_len   >= jjnr_maxlen  ) ||
             (shift_len  >= iinr_maxlen  ) ||
             (types_len  >= types_maxlen ) ||
             (jindex_len != (iinr_len+1) ) ||
             (shift_len  != iinr_len     )  ) {
            fprintf(stderr, "Eronous dataset, EXITING!");
            exit(1);
        }

        // set global sizes
        nri = iinr_len;
        nrj = jjnr_len;
        num_particles = types_len;
    }

    fprintf( stderr, "Datatset characteristics are: (num_particles=%d), (nri=%d), (nrj: %d), (last jindex: %d)\n"
           , num_particles, nri, nrj, jindex[nri]
           );

    // shiftvec: [3*23]f32
    // num_particles = 23178
    // pos     : [3*num_particles]real  
    // faction : [3*num_particles]real
    // charge  : [num_particles]real
    // nbfp : [2*ntype*ntype]
    srand(6);
    const int shiftvec_len = 3*23;
    real *shiftvec = (real*) malloc(shiftvec_len * sizeof(real));
    randomInit(shiftvec, shiftvec_len);

    real *pos = (real*) malloc(3*num_particles * sizeof(real));
    randomInit(pos, 3*num_particles);

    real *faction0 = (real*) malloc(3*num_particles * sizeof(real));
    randomInit(faction0, 3*num_particles);

    //printArray(faction0, 3*num_particles);

    real *faction = (real*) malloc(3*num_particles * sizeof(real));
    for(int i=0; i<3*num_particles; i++) { faction[i] = faction0[i]; }

    real *charge = (real*) malloc(3*num_particles * sizeof(real));
    randomInit(charge, 3*num_particles);
    
    const int nbfp_len = 2 * ntype * ntype;
    real *nbfp = (real*) malloc(nbfp_len * sizeof(real));
    randomInit(nbfp, nbfp_len);

    // 2. golden sequential timing
    const unsigned long int elapsed_seq = 
          inl1100_sequential( nri, nrj, num_particles
                            , jindex, iinr, jjnr, shift, types
                            , ntype, facel
                            , shiftvec, pos, faction, charge, nbfp );
    printf("Sequential version took: %lu microseconds to run!\n", elapsed_seq);

    { // 3. querry the hardware
        int nDevices;
        cudaGetDeviceCount(&nDevices);

        cudaDeviceProp prop;

        cudaGetDeviceProperties(&prop, 0);
        HWD = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
        MAX_BLOCK_SZ = prop.maxThreadsPerBlock;
        SH_MEM_SZ = prop.sharedMemPerBlock;
        if (DEBUG_INFO) {
            printf("Device name: %s\n", prop.name);
            printf("Number of hardware threads: %d\n", HWD);
            printf("Block size: %d\n", MAX_BLOCK_SZ);
            printf("Shared memory size: %d\n", SH_MEM_SZ);
            puts("====");
        }
    }

    int   *jindex_d = NULL, *iinr_d = NULL, *jjnr_d = NULL, *shift_d = NULL, *types_d = NULL;
    real *shiftvec_d = NULL, *pos_d = NULL, *faction_d = NULL, *charge_d = NULL, *nbfp_d = NULL;
    real *faction0_d = NULL;
    { // 4. making the CUDA dataset
        const int mem_jindex = (nri+1)*sizeof(int);
        const int mem_iinr   = nri*sizeof(int);
        const int mem_jjnr   = nrj*sizeof(int);
        const int mem_types  = num_particles*sizeof(int);
        const int mem_shiftvec = shiftvec_len * sizeof(real);
        const int mem_pos    = 3*num_particles*sizeof(real);
        const int mem_nbfp   = nbfp_len * sizeof(real);

        cudaMalloc((void**) &jindex_d, mem_jindex);
        cudaMalloc((void**) &iinr_d,   mem_iinr);
        cudaMalloc((void**) &jjnr_d,   mem_jjnr);
        cudaMalloc((void**) &shift_d,  mem_iinr);
        cudaMalloc((void**) &types_d,  mem_types);

        cudaMalloc((void**) &shiftvec_d,  mem_shiftvec);
        cudaMalloc((void**) &pos_d,       mem_pos);
        cudaMalloc((void**) &faction_d,   mem_pos);
        cudaMalloc((void**) &faction0_d,  mem_pos);
        cudaMalloc((void**) &charge_d,    mem_pos);
        cudaMalloc((void**) &nbfp_d,      mem_nbfp);

        cudaMemcpy(jindex_d, jindex, mem_jindex, cudaMemcpyHostToDevice);
        cudaMemcpy(iinr_d, iinr, mem_iinr, cudaMemcpyHostToDevice);
        cudaMemcpy(jjnr_d, jjnr, mem_jjnr, cudaMemcpyHostToDevice);
        cudaMemcpy(shift_d, shift, mem_iinr, cudaMemcpyHostToDevice);
        cudaMemcpy(types_d, types, mem_types, cudaMemcpyHostToDevice);

        cudaMemcpy(shiftvec_d, shiftvec, mem_shiftvec, cudaMemcpyHostToDevice);
        cudaMemcpy(pos_d, pos, mem_pos, cudaMemcpyHostToDevice);
        cudaMemcpy(faction_d, faction0, mem_pos, cudaMemcpyHostToDevice);
        cudaMemcpy(faction0_d,faction0, mem_pos, cudaMemcpyHostToDevice);
        cudaMemcpy(charge_d, charge, mem_pos, cudaMemcpyHostToDevice);
        cudaMemcpy(nbfp_d, nbfp, mem_nbfp, cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        gpuAssert( cudaPeekAtLastError() );
    }

    { // 5. cuda timing for the version that uses 6 atomic adds per inner iteration
        const int len_flat = jindex[nri];
        const unsigned long int elapsed_cuda_allhist = 
          inl1100_cuda_allhist( len_flat, nri, nrj, num_particles
                              , jindex_d, iinr_d, jjnr_d, shift_d, types_d
                              , ntype, facel
                              , shiftvec_d, pos_d, faction0_d, faction_d
                              , charge_d, nbfp_d, faction );
#if WITH_HDW
        printf("CUDA HWD timing for version using 6 atomic adds per inner iteration: %lu microseconds to run!\n"
              , elapsed_cuda_allhist);
#else
        printf("CUDA CAS timing for version using 6 atomic adds per inner iteration: %lu microseconds to run!\n"
              , elapsed_cuda_allhist);
#endif

        if (argc == 6) {
          printf("Writing runtine to %s\n", argv[5]);
          FILE *f = fopen(argv[5], "w");
          fprintf(f, "%f\n", elapsed_cuda_allhist/1e6);
          fclose(f);
        }
    }

    { // free cuda and hoist memory
        free(jindex); free(iinr); free(jjnr); free(shift); free(types);
        free(shiftvec); free(pos); free(faction0); free(faction);
        free(charge); free(nbfp);

        cudaFree(jindex_d); cudaFree(iinr_d); cudaFree(jjnr_d); cudaFree(shift_d);
        cudaFree(types_d); cudaFree(shiftvec_d); cudaFree(pos_d); 
        cudaFree(faction0_d); cudaFree(faction_d); cudaFree(charge_d); cudaFree(nbfp_d);        
    }
    return 0;
}
