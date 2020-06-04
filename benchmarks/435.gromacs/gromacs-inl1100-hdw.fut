-- ==
-- entry: main
--
-- compiled input @ data/all-huge.in
-- output @ data/all-huge.out

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


--  let (ws,hs) = map f1 bin                         |\label{line:whs-comp}|
--  let B$_w$  = scan$^{exc}$ (+) 0 ws                 |\label{line:Bw}|
--  let len$_{flat}$= B$_w$[q-1] + ws[q-1]
--  let tmp = map2 ($\lambda$s b $\rightarrow$ if s == 0 then -1 else b) ws B$_w$
--  let flag= scatter (replicate len$_{flat}$ 0) tmp (replicate q 1) |\label{line:flgs}|
--  let tmp = scan$^{inc}$ (+) 0 flag                  |\label{line:scn-flgs}|
--  let out$_{inds}$ = map ($\lambda$ x $\rightarrow$ x-1) tmp           |\label{line:oinds}|
--  let tmp = map ($\lambda$ f$\rightarrow$ 1-f) flag  |\label{line:negflg}|
--  let inn$_{inds}$ = sgmscan$^{inc}$ (+) 0 flag tmp        |\label{line:iotaws}|

let sgmscan 't [n] (op: (t -> t -> t)) (ne: t) (flag: [n]i8) (vals: [n]t) : [n]t =
  let (_, res) = scan (\ (f1,v1) (f2,v2) ->
                            let f = f1 | f2
                            let v = if f2 != 0 then v2 else op v1 v2
                            in  (f,v)
                      ) (0, ne) (zip flag vals) |> unzip
  in  res

let reduce_by_index_rf 'a [m] [n] (rf: i32) (dest : *[m]a) (f : a -> a -> a) (ne : a) (is : [n]i32) (as : [n]a) : *[m]a =
  intrinsics.hist (rf, dest, f, ne, is, as) :> *[m]a

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
entry main  [nri] [nrip1] [nrj] [num_particles] [histo_len]
            (jindex: [nrip1]i32) (iinr: [nri]i32) (jjnr: [nrj]i32)
            (shift: [nri]i32) (types: [num_particles]i32)
            (ntype: i32) (facel: real)
            (shiftvec: []real) (pos: [histo_len]real) (faction1d: [histo_len]real)
            (charge: [num_particles]real) (nbfp: []real)
        = --: *[]real =
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
      let tjA               = ntiA + 2 * (types[jnr])
      let vnb6              = rinvsix * (nbfp[tjA])
      let vnb12             = rinvsix * rinvsix * (nbfp[tjA+1])
      let qq                = iqA * (charge[jnr])
      let vcoul             = qq*rinv11
      let fs11              = (twelve*vnb12-six*vnb6+vcoul)*rinvsq11
      let tx11              = dx11*fs11
      let ty11              = dy11*fs11
      let tz11              = dz11*fs11
      in  ([tx11, ty11, tz11], jnr, iinr[oind])

  let faction_iinr = replicate histo_len nul |> unflatten num_particles 3
  let faction_jjnr = copy faction1d |> unflatten num_particles 3
  let (txyz11s2Ds_iinr, ext_jnrs, ext_inrs) = unzip3 <| map2 inner_body out_inds (iota len_flat)
  let txyz11s2Ds_jjnr = map (\row -> map (\x -> nul-x) row) txyz11s2Ds_iinr

  let faction_jjnr' = reduce_by_index_rf 150i32 faction_jjnr (map2 (+)) [nul,nul,nul] ext_jnrs txyz11s2Ds_jjnr |> flatten
  let faction_iinr' = reduce_by_index_rf 150i32 faction_iinr (map2 (+)) [nul,nul,nul] ext_inrs txyz11s2Ds_iinr |> flatten

  let faction' = map (\i-> faction_iinr'[i] + faction_jjnr'[i]) (iota histo_len)
  in  faction'
