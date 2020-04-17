#ifndef GROMACS_WRAPPER
#define GROMACS_WRAPPER

#include "scan-host-skel.cu.h"

/**********************************/
/*** Golden Sequntial Gromacs   ***/
/**********************************/

unsigned long int inl1100_sequential(
        const int nri, const int nrj, const int num_particles,
        int* jindex, int* iinr, int* jjnr, int* shift, int* types,
        const int ntype, const int facel,
        real* shiftvec, real* pos, real* faction0, real* charge, real* nbfp
) {
    unsigned long int elapsed;
    real* faction = (real*)malloc(3*num_particles*sizeof(real));

    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for(int k=0; k<CPU_RUNS; k++) {
        for(int i=0; i < 3*num_particles; i++) {
            faction[i] = faction0[i];
        }
        for(int n=0; n < nri; n++) {
            const int is3        = 3*shift[n];  
            const real shX       = shiftvec[is3];    
            const real shY       = shiftvec[is3+1];  
            const real shZ       = shiftvec[is3+2]; 
            const int ii         = iinr[n]; 
            const int ii3        = 3*ii;         
            const int nj0        = jindex[n];        
            const int nj1        = jindex[n+1];      
            const real ix1       = shX + pos[ii3];   
            const real iy1       = shY + pos[ii3+1]; 
            const real iz1       = shZ + pos[ii3+2];
            const real iqA       = facel*charge[ii];
            const int   ntiA     = 2*ntype*types[ii];
            real fix1            = nul;
            real fiy1            = nul;
            real fiz1            = nul;
            for(int k=nj0; k<nj1; k++) {
                const int jnr          = jjnr[k];
                const int j3           = 3*jnr;
                const real jx1         = pos[j3];
                const real jy1         = pos[j3+1];
                const real jz1         = pos[j3+2];
                const real dx11        = ix1 - jx1;
                const real dy11        = iy1 - jy1;
                const real dz11        = iz1 - jz1;
                const real rsq11       = dx11*dx11+dy11*dy11+dz11*dz11;
                const real rinv11      = one / sqrtf(rsq11);
                const real rinvsq11    = rinv11*rinv11;
                const real rinvsix     = rinvsq11*rinvsq11*rinvsq11;
                const int tjA          = ntiA+2*types[jnr];
                const real vnb6        = rinvsix*nbfp[tjA];
                const real vnb12       = rinvsix*rinvsix*nbfp[tjA+1];
                const real qq          = iqA*charge[jnr];
                const real vcoul       = qq*rinv11;
                const real fs11        = (twelve*vnb12-six*vnb6+vcoul)*rinvsq11;
                const real tx11        = dx11*fs11;
                const real ty11        = dy11*fs11;
                const real tz11        = dz11*fs11;
                fix1                   = fix1 + tx11;
                fiy1                   = fiy1 + ty11;
                fiz1                   = fiz1 + tz11;
                faction[j3]            = faction[j3] - tx11;
                faction[j3+1]          = faction[j3+1]-ty11;
                faction[j3+2]          = faction[j3+2]-tz11;
            }
            faction[ii3]      = faction[ii3]   + fix1;
            faction[ii3+1]    = faction[ii3+1] + fiy1;
            faction[ii3+2]    = faction[ii3+2] + fiz1;
        }
    }
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 

    for(int i=0; i < 3*num_particles; i++) {
        faction0[i] = faction[i];
    }
    free(faction);
    //printf("Sequential Naive version runs in: %lu microsecs\n", elapsed);
    return (elapsed / CPU_RUNS); 
}


/*************************************/
/*** Wrapper for Final Reduce Step ***/
/*************************************/

unsigned long int inl1100_cuda_allhist(
        const int len_flat, const int nri, const int nrj, const int num_particles,
        int* jindex, int* iinr, int* jjnr, int* shift, int* types,
        const int ntype, const int facel,
        real* shiftvec, real* pos, real* faction0, real* faction,
        real* charge, real* nbfp, real* faction_h
) {
    const uint32_t B = 256;
    const unsigned int mem_flag = len_flat * sizeof(char);
    const unsigned int mem_inds = len_flat * sizeof(int32_t);
    const unsigned int mem_tmp_scan = MAX_BLOCK_SZ * sizeof(int32_t);
    const unsigned int mem_tmp_flag = MAX_BLOCK_SZ * sizeof(char);
    const unsigned int mem_ntiAs = nri * sizeof(int32_t);
    const unsigned int mem_ixyz  = nri * sizeof(real);
    const unsigned int mem_faction = 3 * num_particles * sizeof(real);

    char *flag = NULL, *tmp_flag = NULL;
    int32_t *out_inds = NULL, *inn_inds = NULL, *tmp_scan = NULL, *ntiAs = NULL;
    real *ix1s = NULL, *iy1s = NULL, *iz1s = NULL, *iqAs = NULL;
    
    cudaMalloc((void**) &flag,     mem_flag);
    cudaMalloc((void**) &out_inds, mem_inds);
    cudaMalloc((void**) &inn_inds, mem_inds);
    cudaMalloc((void**) &tmp_scan, mem_tmp_scan);
    cudaMalloc((void**) &tmp_flag, mem_tmp_flag);

    cudaMalloc((void**) &ntiAs, mem_ntiAs);
    cudaMalloc((void**) &ix1s,  mem_ixyz);
    cudaMalloc((void**) &iy1s,  mem_ixyz);
    cudaMalloc((void**) &iz1s,  mem_ixyz);
    cudaMalloc((void**) &iqAs,  mem_ixyz);

    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for(int i=0; i < GPU_RUNS; i++) {
        cudaMemset(flag, 0, mem_flag);
        cudaMemcpy(faction, faction0, mem_faction, cudaMemcpyDeviceToDevice);

        { // make flag array
            const uint32_t num_threads = nri;
            const uint32_t num_blocks = (num_threads + B - 1) / B;
            mkFlagKernel<<<num_blocks,B>>>(flag, jindex, nri);
        }
        //gpuAssert( cudaPeekAtLastError() );
        //fprintf(stderr, "1111\n");

        // out_inds = map (\i -> if i==0 then 0 else flag[i]) (iota len_flat) |> scan (+) 0i32
        scanInc<Add1>( B, len_flat, out_inds, flag, tmp_scan );
        //gpuAssert( cudaPeekAtLastError() );
        //fprintf(stderr, "2222, len_flat: %d\n", len_flat);

        // set first element of flag baxck to 1
        setFirstFlagElmKernel<<<1,16>>>(flag);
        //gpuAssert( cudaPeekAtLastError() );
        //fprintf(stderr, "3333\n");

        //let inn_inds = map (\f -> 1-f) flag |> sgmscan (+) 0 flag
        sgmScanInc<Add2> ( B, len_flat, inn_inds, flag, flag, tmp_scan, tmp_flag );
        //gpuAssert( cudaPeekAtLastError() );
        //fprintf(stderr, "4444\n");

        { // outer loop part
            const uint32_t num_threads = nri;
            const uint32_t num_blocks = (num_threads + B - 1) / B;
            outerLoopKernel<<<num_blocks,B>>>( nri, facel, ntype, shift, shiftvec, iinr, types
                                             , pos, charge, ix1s, iy1s, iz1s, iqAs, ntiAs);
        }
        //gpuAssert( cudaPeekAtLastError() );
        //fprintf(stderr, "5555\n");

        { // inner loop part
            const uint32_t num_threads = len_flat;
            const uint32_t num_blocks = (num_threads + B - 1) / B;
            innerLoopKernel<<<num_blocks,B>>>( len_flat, jindex, out_inds, inn_inds
                                             , iinr, jjnr, types, pos, charge, nbfp
                                             , ix1s, iy1s, iz1s, iqAs, ntiAs, faction
                                             );
        }
        //gpuAssert( cudaPeekAtLastError() );
        //fprintf(stderr, "6666\n");
    }

    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    const unsigned long int elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

    { // validate and free memory
        bool is_valid;
        real* faction_dh = (real*)malloc(mem_faction);
        cudaMemcpy(faction_dh, faction, mem_faction, cudaMemcpyDeviceToHost);
        is_valid = validate32(faction_h, faction_dh, 3 * num_particles);
        

        free(faction_dh);
        cudaFree(flag); cudaFree(out_inds); cudaFree(inn_inds);
        cudaFree(tmp_scan); cudaFree(tmp_flag);
        cudaFree(ntiAs); cudaFree(iqAs);
        cudaFree(ix1s); cudaFree(iy1s); cudaFree(iz1s);

        if(!is_valid) {
            fprintf(stderr, "Validation of inl1100_cuda_allhist FAILS! Exiting!\n\n");
            //exit(1);
        }
    }
    
    return (elapsed / GPU_RUNS);
}

/*************************************/
/*** Wrapper for Final Reduce Step ***/
/*************************************/
#if 0
inline void
reduceAcrossMultiHistos(AtomicPrim select, uint32_t H, uint32_t M, uint32_t B, uint32_t* d_histos, uint32_t* d_histo) {
    // reduce across subhistograms
    const size_t num_blocks_red = (H + B - 1) / B;
    if(select == ADD) {
        naive_atmadd_reduce_kernel<<< num_blocks_red, B >>>
                (d_histos, d_histo, H, M);
    } else if (select == CAS) {
        naive_satadd_reduce_kernel<<< num_blocks_red, B >>>
                (d_histos, d_histo, H, M);
    } else {
        naive_argmin_reduce_kernel<<< num_blocks_red, B >>>
                ((uint64_t*)d_histos, (uint64_t*)d_histo, H, M);
    }
}

/********************************/
/*** Global-Memory Histograms ***/
/********************************/
unsigned long
glbMemHwdAddCoop(AtomicPrim select, const int RF, const int N, const int H, const int B, const int M, const int num_chunks, int* d_input, uint32_t* h_ref_histo) {
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
    const size_t K = (select == XCHG) ? 2 : 1;
    const size_t mem_size_histo  = H * K * sizeof(uint32_t);
    const size_t mem_size_histos = M * mem_size_histo;
    const size_t mem_size_locks  = M * H * sizeof(uint32_t);
    uint32_t* d_histos;
    uint32_t* d_histo;
    int* d_locks;
    uint32_t* h_histo = (uint32_t*)malloc(mem_size_histo);

    cudaMalloc((void**) &d_histos, mem_size_histos);
    cudaMalloc((void**) &d_histo,  mem_size_histo );
    cudaMalloc((void**) &d_locks,  mem_size_locks);
    cudaMemset(d_locks,  0, mem_size_locks );
    cudaMemset(d_histo,  0, mem_size_histo );
    cudaMemset(d_histos, 0, mem_size_histos);
    cudaDeviceSynchronize();

    { // dry run
      for(int k=0; k<num_chunks; k++) {
        if(select == ADD) {
          glbMemHwdAddCoopKernel<ADD, uint32_t><<< num_blocks, B >>>
              (RF, N, H, M, T, k*chunk_size, (k+1)*chunk_size, d_input, d_histos, NULL);
        } else if (select == CAS){
          glbMemHwdAddCoopKernel<CAS, uint32_t><<< num_blocks, B >>>
              (RF, N, H, M, T, k*chunk_size, (k+1)*chunk_size, d_input, d_histos, NULL);
        } else { // select == XCHG
          glbMemHwdAddCoopKernel<XCHG,uint64_t><<< num_blocks, B >>>
              (RF, N, H, M, T, k*chunk_size, (k+1)*chunk_size, d_input, (uint64_t*)d_histos, d_locks);
        }
      }
      // reduce across subhistograms
      reduceAcrossMultiHistos(select, H, M, B, d_histos, d_histo);
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    const int num_gpu_runs = GPU_RUNS;

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 

    for(int q=0; q<num_gpu_runs; q++) {
      cudaMemset(d_histos, 0, mem_size_histos);

      for(int k=0; k<num_chunks; k++) {
        if(select == ADD) {
          glbMemHwdAddCoopKernel<ADD, uint32_t><<< num_blocks, B >>>
              (RF, N, H, M, T, k*chunk_size, (k+1)*chunk_size, d_input, d_histos, NULL);
        } else if (select == CAS){
          glbMemHwdAddCoopKernel<CAS, uint32_t><<< num_blocks, B >>>
              (RF, N, H, M, T, k*chunk_size, (k+1)*chunk_size, d_input, d_histos, NULL);
        } else { // select == XCHG
          glbMemHwdAddCoopKernel<XCHG,uint64_t><<< num_blocks, B >>>
              (RF, N, H, M, T, k*chunk_size, (k+1)*chunk_size, d_input, (uint64_t*)d_histos, d_locks);
        }
      }
      // reduce across subhistograms
      reduceAcrossMultiHistos(select, H, M, B, d_histos, d_histo);
    }
    cudaDeviceSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    gpuAssert( cudaPeekAtLastError() );

    { // reduce across histograms and copy to host
        cudaMemcpy(h_histo, d_histo, mem_size_histo, cudaMemcpyDeviceToHost);
    }

    { // validate and free memory
        bool is_valid;

        if (select == XCHG) {
            is_valid = validate64((uint64_t*)h_histo, (uint64_t*)h_ref_histo, H); 
        } else {
            is_valid = validate32(h_histo, h_ref_histo, H);
        }

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
#endif

#endif // GROMACS_WRAPPER
