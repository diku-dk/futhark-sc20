#ifndef HISTO_WRAPPER
#define HISTO_WRAPPER

/***********************/
/*** Various Helpers ***/
/***********************/
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}
void randomInit(int* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand(); // (float)RAND_MAX;
}
void zeroOut(int* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = 0;
}
template<class T>
bool validate(T* A, T* B, unsigned int sizeAB) {
    for(int i = 0; i < sizeAB; i++) {
        if (A[i] != B[i]) {
            printf("INVALID RESULT %d %d %d\n", i, A[i], B[i]);
            return false;
        }
    }
    //printf("VALID RESULT!\n");
    return true;
}
int gpuAssert(cudaError_t code) {
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}
/**********************************/
/*** Golden Sequntial Histogram ***/
/**********************************/
void computeSeqHisto(const int N, const int H, int* input, int* histo) {
    for(int i=0; i<N; i++) {
        struct indval<int> iv = f<int>(input[i], H);
        histo[iv.index] += iv.value;
    }
}

unsigned long
goldSeqHisto(const int N, const int H, int* input, int* histo) {
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for(int q=0; q<CPU_RUNS; q++) {
        zeroOut(histo, H);
        computeSeqHisto(N, H, input, histo);
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    //printf("Sequential Naive version runs in: %lu microsecs\n", elapsed);
    return (elapsed / CPU_RUNS); 
}

/*******************************/
/*** Local-Memory Histograms ***/
/*******************************/
unsigned long
locMemHwdAddCoop(AtomicPrim select, const int N, const int H, const int histos_per_block
                , const int num_chunks, int* d_input, int* h_ref_histo) {
    if(histos_per_block <= 0) {
        printf("Illegal subhistogram degree: %d, H:%d, XCG?=%d, EXITING!\n", histos_per_block, H, (select==XCHG));
        exit(0);
    }

    // setup execution parameters
    const int    Hchunk = (H + num_chunks - 1) / num_chunks;
    const size_t num_blocks = (NUM_THREADS(N) + BLOCK - 1) / BLOCK; 
    const size_t shmem_size = histos_per_block * Hchunk * sizeof(int);

    printf( "Running Subhistogram degree: %d, num-chunks: %d, H: %d, Hchunk: %d, XCG?= %d, shmem: %ld\n"
          , histos_per_block, num_chunks, H, Hchunk, (select==XCHG), shmem_size );

    const size_t mem_size_histo  = H * sizeof(int);
    const size_t mem_size_histos = histos_per_block * num_blocks * mem_size_histo;
    int* d_histos;
    int* d_histo;
    int* h_histo = (int*)malloc(H*sizeof(int));
    cudaMalloc((void**) &d_histos, mem_size_histos);
    cudaMalloc((void**) &d_histo,  H * sizeof(int));

    { // dry run
      for(int k=0; k<num_chunks; k++) {
        const int chunkLB = k*Hchunk;
        const int chunkUB = min(H, (k+1)*Hchunk);
        if(select == ADD) {
          locMemHwdAddCoopKernel<ADD><<< num_blocks, BLOCK, shmem_size >>>
              (N, H, histos_per_block, chunkLB, chunkUB, NUM_THREADS(N), d_input, d_histos);
        } else if (select == CAS){
          locMemHwdAddCoopKernel<CAS><<< num_blocks, BLOCK, shmem_size >>>
              (N, H, histos_per_block, chunkLB, chunkUB, NUM_THREADS(N), d_input, d_histos);
        } else { // select == XCHG
          locMemHwdAddCoopKernel<XCHG><<< num_blocks, BLOCK, 2*shmem_size >>>
              (N, H, histos_per_block, chunkLB, chunkUB, NUM_THREADS(N), d_input, d_histos);
        }
      }
    }
    cudaThreadSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    cudaMemset(d_histos, 0, mem_size_histos);
    cudaMemset(d_histo , 0, mem_size_histo );

    const int num_gpu_runs = GPU_RUNS;
                             //(((select==XCHG) || (select==CAS)) && (histos_per_block==1)) ?
                             //max(1, GPU_RUNS/25) : GPU_RUNS;

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 

    for(int q=0; q<num_gpu_runs; q++) {
      for(int k=0; k<num_chunks; k++) {
        const int chunkLB = k*Hchunk;
        const int chunkUB = min(H, (k+1)*Hchunk);

        if(select == ADD) {
          locMemHwdAddCoopKernel<ADD><<< num_blocks, BLOCK, shmem_size >>>
              (N, H, histos_per_block, chunkLB, chunkUB, NUM_THREADS(N), d_input, d_histos);
        } else if (select == CAS){
          locMemHwdAddCoopKernel<CAS><<< num_blocks, BLOCK, shmem_size >>>
              (N, H, histos_per_block, chunkLB, chunkUB, NUM_THREADS(N), d_input, d_histos);
        } else { // select == XCHG
          locMemHwdAddCoopKernel<XCHG><<< num_blocks, BLOCK, 2*shmem_size >>>
              (N, H, histos_per_block, chunkLB, chunkUB, NUM_THREADS(N), d_input, d_histos);
        }
      }
    }
    cudaThreadSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    gpuAssert( cudaPeekAtLastError() );

    { // reduce across histograms and copy to host
        const size_t num_blocks_red = (H + BLOCK - 1) / BLOCK;
        naive_reduce_kernel<<< num_blocks_red, BLOCK >>>(d_histos, d_histo, H, histos_per_block*num_blocks);
        cudaMemcpy(h_histo, d_histo, mem_size_histo, cudaMemcpyDeviceToHost);
    }

    { // validate and free memory
        bool is_valid = validate<int>(h_histo, h_ref_histo, H);

        free(h_histo);
        cudaFree(d_histos);
        cudaFree(d_histo);

        if(!is_valid) {
            int coop = (BLOCK + histos_per_block - 1) / histos_per_block;
            printf( "locMemHwdAddCoop: Validation FAILS! M:%d, coop:%d, H:%d, ADD:%d, Exiting!\n\n"
                  , histos_per_block, coop, H, (int)select );
            exit(1);
        }
    }

    return (elapsed/num_gpu_runs);
}

/********************************/
/*** Global-Memory Histograms ***/
/********************************/
unsigned long
glbMemHwdAddCoop(AtomicPrim select, const int N, const int H, const int B, const int M, const int num_chunks, int* d_input, int* h_ref_histo) {
    const int T = NUM_THREADS(N);
    const int C = (T + M - 1) / M;
    const int chunk_size = (H + num_chunks - 1) / num_chunks;

#if 0
    const int C = min( T, (int) ceil(H / k) );
    const int M = (T+C-1) / C;
#endif

    if((C <= 0) || (C > T)) {
        printf("Illegal subhistogram degree M: %d, resulting in C:%d for H:%d, XCG?=%d, EXITING!\n", M, C, H, (select==XCHG));
        exit(0);
    }
    
    // setup execution parameters
    const size_t num_blocks = (T + B - 1) / B;
    const size_t mem_size_histo  = H * sizeof(int);
    const size_t mem_size_histos = M * mem_size_histo;
    const size_t mem_size_locks  = mem_size_histos;
    int* d_histos;
    int* d_histo;
    int* d_locks;
    int* h_histo = (int*)malloc(H*sizeof(int));

    cudaMalloc((void**) &d_histos, mem_size_histos);
    cudaMalloc((void**) &d_histo,  mem_size_histo );
    cudaMalloc((void**) &d_locks,  mem_size_histos);
    cudaMemset(d_locks,  0, mem_size_locks );
    cudaMemset(d_histo,  0, mem_size_histo );
    cudaMemset(d_histos, 0, mem_size_histos);

    { // dry run
      for(int k=0; k<num_chunks; k++) {
        if(select == ADD) {
          glbMemHwdAddCoopKernel<ADD><<< num_blocks, B >>>
              (N, H, M, T, k*chunk_size, (k+1)*chunk_size, d_input, d_histos, NULL);
        } else if (select == CAS){
          glbMemHwdAddCoopKernel<CAS><<< num_blocks, B >>>
              (N, H, M, T, k*chunk_size, (k+1)*chunk_size, d_input, d_histos, NULL);
        } else { // select == XCHG
          glbMemHwdAddCoopKernel<XCHG><<< num_blocks, B >>>
              (N, H, M, T, k*chunk_size, (k+1)*chunk_size, d_input, d_histos, d_locks);
        }
      }
    }
    cudaThreadSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    const int num_gpu_runs = GPU_RUNS;

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 

    for(int q=0; q<num_gpu_runs; q++) {
      cudaMemset(d_histos, 0, mem_size_histos);
      for(int k=0; k<num_chunks; k++) {
        if(select == ADD) {
          glbMemHwdAddCoopKernel<ADD><<< num_blocks, B >>>
              (N, H, M, T, k*chunk_size, (k+1)*chunk_size, d_input, d_histos, NULL);
        } else if (select == CAS){
          glbMemHwdAddCoopKernel<CAS><<< num_blocks, B >>>
              (N, H, M, T, k*chunk_size, (k+1)*chunk_size, d_input, d_histos, NULL);
        } else { // select == XCHG
          glbMemHwdAddCoopKernel<XCHG><<< num_blocks, B >>>
              (N, H, M, T, k*chunk_size, (k+1)*chunk_size, d_input, d_histos, d_locks);
        }
      }
      // reduce across subhistograms
      naive_reduce_kernel<<< (H+B-1) / B, BLOCK >>>(d_histos, d_histo, H, M);
    }
    cudaThreadSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    gpuAssert( cudaPeekAtLastError() );

    { // reduce across histograms and copy to host
        //const size_t num_blocks_red = (H + B - 1) / B;
        //naive_reduce_kernel<<< num_blocks_red, BLOCK >>>(d_histos, d_histo, H, M);
        cudaMemcpy(h_histo, d_histo, mem_size_histo, cudaMemcpyDeviceToHost);
    }

    { // validate and free memory
        bool is_valid = validate<int>(h_histo, h_ref_histo, H);

        free(h_histo);
        cudaFree(d_histos);
        cudaFree(d_histo);
        cudaFree(d_locks);

        if(!is_valid) {
            printf( "glbMemHwdAddCoop: Validation FAILS! B:%d, T:%d, N:%d, H:%d, M:%d, coop:%d, XCHG:%d, Exiting!\n\n"
                  , B, T, N, H, M, C, (int)(select==XCHG) );
            exit(1);
        }
    }

    return (elapsed/num_gpu_runs);
}

/***********************/
/*** Pretty printing ***/
/***********************/
template<int num_histos, int num_m_degs>
void printTextTab( const unsigned long runtimes[3][num_histos][num_m_degs]
                 , const int histo_sizes[num_histos]
                 , const int kms[num_m_degs]
                 , const int R
) {
    for(int k=0; k<3; k++) {
        if     (k==0) printf("ADD, R=%d\t", R);
        else if(k==1) printf("CAS, R=%d\t", R);
        else if(k==2) printf("XCG, R=%d\t", R);

        for(int i = 0; i<num_histos; i++) { printf("H=%d\t", histo_sizes[i]); }
        printf("\n");
        for(int j=0; j<num_m_degs; j++) {
            if      (j==0)             printf("M=1 \t");
            else if (j < num_m_degs-1) printf("M=%d\t", kms[j]);
            else                       printf("Ours\t");
            for(int i = 0; i<num_histos; i++) {
                printf("%lu\t", runtimes[k][i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
template<int num_histos, int num_m_degs>
void printLaTex( const unsigned long runtimes[3][num_histos][num_m_degs]
               , const int histo_sizes[num_histos]
               , const int kms[num_m_degs]
               , const int R) {
    for(int k=0; k<3; k++) {
        printf("\\begin{tabular}{|l|l|l|l|l|l|}\\hline\n");
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
                printf("\t& %lu", runtimes[k][i][j]);
            }
            printf("\\\\");
            if(j == (num_m_degs-1)) printf("\\hline");
            printf("\n");
        }
        printf("\\end{tabular}\n");
    }
}
#endif // HISTO_WRAPPER
