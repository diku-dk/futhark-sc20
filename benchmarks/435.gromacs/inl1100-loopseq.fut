-- ==
-- entry: main
--
-- compiled input @ data/all-huge-correct.in
-- output @ data/all-huge-correct.out
--
-- compiled input @ data/all-largest.in
-- output @ data/all-largest.out

-------------------------------------------------------------
--   * This is gromacs innerloop inl1100
--   * Forces:      Calculated
--   * Coulomb:     Normal
--   * Nonbonded:   Lennard-Jones
--   * Solvent opt: No
--   * Free energy: No
--------------------------------------------------------------

type real   =  f32
let nul     =  0.0f32
let one     =  1.0f32
let six     =  6.0f32
let twelve  = 12.0f32


-- ntype 19, types \in [0..18]
-- shiftvec: [3*23]f32
-- num_particles = 23178
-- pos     : [3*num_particles]real
-- faction : [3*num_particles]real
-- charge  : [num_particles]real
-- nbfp : [2*ntype*ntype]
entry main  [nri] [nrip1] [nrj] [num_particles]
            (jindex: [nrip1]i32) (iinr: [nri]i32) (jjnr: [nrj]i32)
            (shift: [nri]i32) (types: [num_particles]i32) 
            (ntype: i32) (facel: real)
            (shiftvec: []real) (pos: []real) (faction: []real)
            (charge: []real) (nbfp: []real) 
        = --: *[]real =

  let faction' = 
    loop (faction) = (copy faction)
    for n < nri do
        let is3               = 3*shift[n]       -- temporary
        let shX               = shiftvec[is3]    -- temporary
        let shY               = shiftvec[is3+1]  -- temporary
        let shZ               = shiftvec[is3+2]  -- temporary
        let ii                = iinr[n]          -- temporary
        let ii3               = 3*ii             -- cheap to recompute 3*ii
        let nj0               = jindex[n]        -- already in jindex
        let nj1               = jindex[n+1]      
        let ix1               = shX + pos[ii3]   -- save
        let iy1               = shY + pos[ii3+1] -- save
        let iz1               = shZ + pos[ii3+2] -- save
        let iqA               = facel*charge[ii] -- save
        let ntiA              = 2*ntype*types[ii] -- save
        let (faction, fix1, fiy1, fiz1) =
            loop (faction, fix1, fiy1, fiz1) = (faction, nul, nul, nul)
            for k0 < (nj1 - nj0) do
              let k                 = k0 + nj0
              let jnr               = jjnr[k]
              let j3                = 3*jnr
              let jx1               = pos[j3]
              let jy1               = pos[j3+1]
              let jz1               = pos[j3+2]
              let dx11              = ix1 - jx1
              let dy11              = iy1 - jy1
              let dz11              = iz1 - jz1
              let rsq11             = dx11*dx11+dy11*dy11+dz11*dz11
              let rinv11            = one / (f32.sqrt rsq11)
              let rinvsq11          = rinv11*rinv11
              let rinvsix           = rinvsq11*rinvsq11*rinvsq11
              let tjA               = ntiA+2*types[jnr]
              let vnb6              = rinvsix*nbfp[tjA]
              let vnb12             = rinvsix*rinvsix*nbfp[tjA+1]
              let qq                = iqA*charge[jnr]
              let vcoul             = qq*rinv11
              let fs11              = (twelve*vnb12-six*vnb6+vcoul)*rinvsq11
              let tx11              = dx11*fs11
              let ty11              = dy11*fs11
              let tz11              = dz11*fs11
              let fix1              = fix1 + tx11
              let fiy1              = fiy1 + ty11
              let fiz1              = fiz1 + tz11
              let faction[j3]       = faction[j3] - tx11
              let faction[j3+1]     = faction[j3+1]-ty11
              let faction[j3+2]     = faction[j3+2]-tz11
              in  (faction, fix1, fiy1, fiz1)
        let faction[ii3]      = faction[ii3] + fix1
        let faction[ii3+1]    = faction[ii3+1] + fiy1
        let faction[ii3+2]    = faction[ii3+2] + fiz1
        in  faction

  in faction'

--futhark bench --backend=opencl --pass-option=--default-num-groups=144 --pass-option=--default-group-size=256 -r 1000 inl1100.fut
