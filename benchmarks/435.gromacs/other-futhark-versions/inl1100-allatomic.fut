-- ==
-- entry: main
--
-- compiled input @ ../data/all-huge.in
-- output @ ../data/all-huge.out

--
-- compiled input @ ../data/all-largest.in
-- output @ ../data/all-largest.out

import "util"

entry main  [nri] [nrip1] [nrj] [num_particles]
            (jindex: [nrip1]i32) (iinr: [nri]i32) (jjnr: [nrj]i32)
            (shift: [nri]i32) (types: [num_particles]i32) 
            (ntype: i32) (facel: real)
            (shiftvec: []real) (pos: []real) (faction: []real)
            (charge: []real) (nbfp: []real) 
        = --: *[]real =

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
      in  ([tx11, ty11, tz11], jnr, iinr[oind])

  let (txyz11s2Ds, ext_jnrs, ext_inrs) = unzip3 <| map2 inner_body out_inds (iota len_flat)

  let len_flat_histo = 6*len_flat
  let (hist_inds, hist_vals) = unzip <|
    map (\j6 ->
            let len3 = 3*len_flat
            let j3 = if j6 < len3 then j6 else j6-len3
            let j  = j3 / 3
            let r  = j3 - 3*j
            let txyz_val = txyz11s2Ds[j,r]
            let (ind, hist_val) = if j6 < len3
                                  then (ext_jnrs[j], nul - txyz_val)
                                  else (ext_inrs[j],       txyz_val)
            let hist_ind = 3*ind + r
            in  (hist_ind, hist_val)
        ) (iota len_flat_histo)

  let faction' = reduce_by_index_rf 79i32 (copy faction) (+) nul hist_inds hist_vals
  in  faction'
  --in (len_flat_histo, num_particles*3, length faction')

--futhark bench --backend=opencl --pass-option=--default-num-groups=144 --pass-option=--default-group-size=256 -r 1000 inl1100.fut
