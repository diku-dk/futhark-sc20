#ifndef GROMACS_WRAPPER
#define GROMACS_WRAPPER

/**********************************/
/*** Golden Sequntial Gromacs   ***/
/**********************************/

unsigned long int inl1100_sequential(
        const int nri, const int nrj, const int num_particles,
        int* jindex, int* iinr, int* jjnr, int* shift, int* types,
        const int ntype, const int facel,
        real* shiftvec, real* pos, real* faction0, real* faction, real* charge, real* nbfp
) {
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
    const unsigned long int elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 

    //printf("Sequential Naive version runs in: %lu microsecs\n", elapsed);
    return (elapsed / CPU_RUNS); 
}


/*************************************/
/*** Wrapper for Final Reduce Step ***/
/*************************************/

unsigned long int inl1100_cuda_allhist() {
    const size_t localWorkSize = 256;
    cl_int ciErr1 = CL_SUCCESS;

    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for(int i=0; i < GPU_RUNS; i++) {
        //gpuToGpuTransfer();
        initKernelParams();
#if 1
        { // memcpy kernel
            const size_t num_threads = 3*buffs.num_particles;
            const size_t num_blocks = (num_threads + localWorkSize - 1) / localWorkSize;
            const size_t globalWorkSize = num_blocks * localWorkSize;

            ciErr1 |= clEnqueueNDRangeKernel(ctrl.queue, kers.memcpy_ker, 1, NULL,
                                                 &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        }
#endif
        { // outer loop part
            const size_t num_threads = buffs.nri;
            const size_t num_blocks = (num_threads + localWorkSize - 1) / localWorkSize;
            const size_t globalWorkSize = num_blocks * localWorkSize;

            ciErr1 |= clEnqueueNDRangeKernel(ctrl.queue, kers.outer_loop_ker, 1, NULL,
                                                 &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        }

        { // inner loop part
            const size_t num_threads = buffs.jindex[buffs.nri];
            const uint32_t num_blocks = (num_threads + localWorkSize - 1) / localWorkSize;
            const size_t globalWorkSize = num_blocks * localWorkSize;

            ciErr1 |= clEnqueueNDRangeKernel(ctrl.queue, kers.inner_loop_ker, 1, NULL,
                                                 &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        }
    }

    clFinish(ctrl.queue);
    OPENCL_SUCCEED(ciErr1);


    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    const unsigned long int elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

    { // validate and free memory
        bool is_valid;
        gpuToCpuTransfer();
        is_valid = validate32(buffs.faction, buffs.faction_dh, 3 * buffs.num_particles);
        
        if(!is_valid) {
            fprintf(stderr, "Validation of inl1100_opencl FAILS! Exiting!\n\n");
            //exit(1);
        }

        printf("\nFaction-Host:\n");
        printArray(buffs.faction, 20);
        printf("\n");
    
        printf("\nFaction-Device:\n");
        printArray(buffs.faction_dh, 20);
        printf("\n");

        printf("\nFaction-Orig:\n");
        printArray(buffs.faction0, 20);
        printf("\n");

    }
    
    return (elapsed / GPU_RUNS);
}
#endif // GROMACS_WRAPPER
