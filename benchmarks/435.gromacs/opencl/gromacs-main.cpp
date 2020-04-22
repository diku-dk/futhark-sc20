#include <math.h>
//#include <sys/time.h>
//#include <time.h>

#define MIN(a,b)    (((a) < (b)) ? (a) : (b))
#define MAX(a,b)    (((a) < (b)) ? (b) : (a))
 
#ifndef DEBUG_INFO
#define DEBUG_INFO  0
#endif

#ifndef LOCMEMW_PERTHD
#define LOCMEMW_PERTHD 12
#endif

#define CPU_RUNS 10
#define GPU_RUNS 1000


unsigned int HWD;
unsigned int SH_MEM_SZ;
unsigned int MAX_BLOCK_SZ;

#define lgWARP      5
#define WARP        (1<<lgWARP)

//#define NUM_BLOCKS_SCAN     1024
#define ELEMS_PER_THREAD    11//12

#define NUM_THREADS(n)  min(n, HWD)

typedef float real;
const real nul     =  0.0;
const real one     =  1.0;
const real six     =  6.0;
const real twelve  = 12.0;

#include "clutils.h"
#include "util.h"
#include "SetupOpenCL.h"
#include "gromacs-wrap.h"

/***********************/
/*** Pretty printing ***/
/***********************/


void usage(const char *prog) {
    fprintf(stderr, "Usage: %s <input-file> <nri_max> <nrj_max> <num_particles_max>\n", prog);
    exit(1);
}


/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////

// nvcc -O4 -arch=compute_35 -prec-sqrt=true gromacs-main.cu
// ./a.out data/huge-indarr.txt.in 36000 1600000 32000
int main(int argc, char **argv) {
    if (argc != 5) {
        usage(argv[0]);
    }

    // 1. reading/constructing the dataset


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
        buffs.jindex = (int*) malloc(jindex_maxlen * sizeof(int));
        buffs.iinr   = (int*) malloc(iinr_maxlen   * sizeof(int));
        buffs.jjnr   = (int*) malloc(jjnr_maxlen   * sizeof(int));
        buffs.shift  = (int*) malloc(iinr_maxlen   * sizeof(int));
        buffs.types  = (int*) malloc(types_maxlen  * sizeof(int));
        readDataset ( file_name
                    , buffs.jindex, jindex_maxlen, &jindex_len
                    , buffs.iinr,   iinr_maxlen,   &iinr_len
                    , buffs.jjnr,   jjnr_maxlen,   &jjnr_len
                    , buffs.shift,  iinr_maxlen,   &shift_len
                    , buffs.types,  types_maxlen,  &types_len
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
        buffs.nri = iinr_len;
        buffs.nrj = jjnr_len;
        buffs.num_particles = types_len;

        buffs.ntype = 19;
        buffs.facel = 0.5;
    }

    fprintf( stderr, "Datatset characteristics are: (num_particles=%d), (nri=%d), (nrj: %d) (last jindex: %d)\n"
           , buffs.num_particles, buffs.nri, buffs.nrj, buffs.jindex[buffs.nri]
           );

    // shiftvec: [3*23]f32
    // num_particles = 23178
    // pos     : [3*num_particles]real  
    // faction : [3*num_particles]real
    // charge  : [num_particles]real
    // nbfp : [2*ntype*ntype]
    srand(6); //srand(2006);
    buffs.shiftvec_len = 3*23;
    buffs.shiftvec = (real*) malloc(buffs.shiftvec_len * sizeof(real));
    randomInit(buffs.shiftvec, buffs.shiftvec_len);

    buffs.pos = (real*) malloc(3*buffs.num_particles * sizeof(real));
    randomInit(buffs.pos, 3*buffs.num_particles);

    buffs.faction0 = (real*) malloc(3*buffs.num_particles * sizeof(real));
    randomInit(buffs.faction0, 3*buffs.num_particles);

    //printArray(faction0, 3*num_particles);

    buffs.faction = (real*) malloc(3*buffs.num_particles * sizeof(real));
    for(int i=0; i<3*buffs.num_particles; i++) { buffs.faction[i] = buffs.faction0[i]; }

    buffs.faction_dh = (real*) malloc(3*buffs.num_particles * sizeof(real));

    buffs.charge = (real*) malloc(3*buffs.num_particles * sizeof(real));
    randomInit(buffs.charge, 3*buffs.num_particles);
    
    buffs.nbfp_len = 2 * buffs.ntype * buffs.ntype;
    buffs.nbfp = (real*) malloc(buffs.nbfp_len * sizeof(real));
    randomInit(buffs.nbfp, buffs.nbfp_len);

    // 2. golden sequential timing
    const unsigned long int elapsed_seq = 
          inl1100_sequential( buffs.nri, buffs.nrj, buffs.num_particles
                            , buffs.jindex, buffs.iinr, buffs.jjnr, buffs.shift, buffs.types
                            , buffs.ntype, buffs.facel
                            , buffs.shiftvec, buffs.pos, buffs.faction0, buffs.faction, buffs.charge, buffs.nbfp );
    printf("Sequential version took: %lu microseconds to run!\n", elapsed_seq);

    { // 3. querry the hardware
        MAX_BLOCK_SZ = 1024;
        SH_MEM_SZ    = 48*1024;
        HWD          = MAX_BLOCK_SZ * 36;

        if (DEBUG_INFO) {
            printf("Number of hardware threads: %d\n", HWD);
            printf("Block size: %d\n", MAX_BLOCK_SZ);
            printf("Shared memory size: %d\n", SH_MEM_SZ);
            puts("====");
        }
    }

    initOclControl();
    initKernels();
    initOclBuffers();    
    initKernelParams();
    
    { // 5. cuda timing for the version that uses 6 atomic adds per inner iteration
        const int len_flat = buffs.jindex[buffs.nri];
        const unsigned long int elapsed_cuda_allhist = 
          inl1100_cuda_allhist();

        printf("CUDA timing for version using 6 atomic adds per inner iteration: %lu microseconds to run!\n"
              , elapsed_cuda_allhist);
    }

    freeBuffers();
    freeOclControl();

    return 0;
}
