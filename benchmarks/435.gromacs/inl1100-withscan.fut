-- ==
-- entry: main
--
-- compiled input @ data/all-huge.in
-- output @ data/all-huge.out

--
-- compiled input @ data/all-largest.in
-- output @ data/all-largest.out

import "util"

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
            (charge: []real) (nbfp: []real) : []real =
  #[unsafe]
  -- building helper structures for flattening!
  let len_flat = jindex[nri]
  let flag= scatter (replicate len_flat 0i8) (jindex[:nri]) (replicate nri 1i8)
  let out_inds = map (\i -> if i==0 then 0i32 else (i32.i8 flag[i])) (iota len_flat)
              |> scan (+) 0i32

  let (ix1s, iy1s, iz1s, iqAs, ntiAs) = unzip5 <|
    map (\n ->
            let is3               = 3*shift[n]       -- temporary
            let shX               = shiftvec[is3]    -- temporary
            let shY               = shiftvec[is3+1]  -- temporary
            let shZ               = shiftvec[is3+2]  -- temporary
            let ii                = iinr[n]          -- temporary
            let ii3               = 3*ii             -- cheap to recompute 3*ii
            --let nj0               = jindex[n]      -- already in jindex
            --let nj1               = jindex[n+1]
            let ix1               = shX + pos[ii3]   -- save
            let iy1               = shY + pos[ii3+1] -- save
            let iz1               = shZ + pos[ii3+2] -- save
            let iqA               = facel*charge[ii] -- save
            let ntiA              = 2*ntype*types[ii] -- save
            in  (ix1, iy1, iz1, iqA, ntiA)
        ) (iota nri)

  let inner_body (oind: i32) (k: i32) =
      let (ix1, iy1, iz1)   = (ix1s[oind], iy1s[oind], iz1s[oind])
      let (iqA, ntiA)       = (iqAs[oind], ntiAs[oind])

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
      in  ((tx11, ty11, tz11), jnr)


  let (txyz11s2Ds, ext_jnrs) = unzip <| map2 inner_body out_inds (iota len_flat)

  let H = num_particles
  let RF = ( calcRF (ext_jnrs[15*H: 16*H]) + calcRF (ext_jnrs[30*H: 31*H]) ) / 2 + 1

  let scaned_txyzs = sgmscan (\ (a1,b1,c1) (a2,b2,c2) -> (a1+a2, b1+b2, c1+c2))
                             (nul, nul, nul) flag txyz11s2Ds
  let (fix1s, fiy1s, fiz1s) = unzip3 <| map (\ b -> scaned_txyzs[b-1] ) jindex[1:]

  -- now the histogram computation
  let (tx11s, ty11s, tz11s) = unzip3 txyz11s2Ds

  let len_flat_histo = 3*len_flat + 3*nri
  let (hist_inds, hist_vals) = unzip <|
    map (\j3 ->
            let j = j3 / 3
            let r = j3 - 3*j in
            if j < len_flat
            then -- get from - txyz11s1Ds
                 let ind = 3*ext_jnrs[j] + r
                 let vla = nul - (if r==0 then tx11s[j] else if r==1 then ty11s[j] else tz11s[j])
                 in  (ind, vla)
            else -- get from fixyz1
                 let n = j - len_flat
                 let ind = 3*iinr[n] + r
                 let vla = if r==0 then fix1s[n] else if r==1 then fiy1s[n] else fiz1s[n]
                 in  (ind, vla)
        ) (iota len_flat_histo)

  let faction' = reduce_by_index_rf 79i32 (copy faction) (+) nul hist_inds hist_vals
  in  faction'
