let reduce_by_index_rf 'a [m] [n] (rf: i32) (dest : *[m]a) (f : a -> a -> a) (ne : a) (is : [n]i32) (as : [n]a) : *[m]a =
  intrinsics.hist (rf, dest, f, ne, is, as) :> *[m]a

-- ==
-- entry: mkImgRegHisto
--
-- input  @ data/large.in
-- output @ data/large.out
-- input  @ data/small.in


entry mkImgRegHisto [n] (dataR: [n]f32) (vals: [n]f32) : (i32, i32, []f32, []f32, []f32) =
    let lb1 = reduce_comm f32.min f32.highest dataR
    let ub1 = reduce_comm f32.max f32.lowest  dataR

    let lb2 = reduce_comm f32.min f32.highest vals
    let ub2 = reduce_comm f32.max f32.lowest  vals

    let h1 = 4 + i32.f32 (ub1 - lb1)
    let h2 = 4 + i32.f32 (ub2 - lb2)
    let hc = h1 * h2

    let det = replicate n 1.0f32

    let ff d lb x k =
            let t  = x - (f32.floor x)
            let t2 = t  * t
            let t3 = t2 * t
            let (a3, a2, a1, a0) =
                if k==0 then (-1.0f32, 3.0f32, -3.0f32, 1.0f32) else
                if k==1 then (3.0f32, -6.0f32,  0.0f32, 4.0f32) else
                if k==2 then (-3.0f32, 3.0f32,  3.0f32, 1.0f32)
                else         (1.0f32,  0.0f32,  0.0f32, 0.0f32)
            let valu = a3*t3 + a2*t2 + a1*t + a0
            let idx = k + ( i32.f32 (x - lb) )
            in  (idx, d * valu / 6.0f32)

    let hc_inp = map(\ ijk ->
                        let i  = ijk >> 4
                        let jk = ijk & 15
                        let j  = jk  >> 2
                        let k  = jk  &  3

                        let d = det[i]
                        let x1 = dataR[i]
                        let (i1, v1) = ff d lb1 x1 j

                        let x2 = vals[i]
                        let (i2, v2) = ff d lb2 x2 k

                        in  (h1*i2 + i1, v2 * v1 / d)
                    ) (iota (16*n))


    let (hc_inds, hc_vals) = unzip hc_inp
    let histC = reduce_by_index_rf 1i32 (replicate hc 0.0f32) (+) 0.0f32 hc_inds hc_vals

    let histC' = unflatten h2 h1 histC
    let hist2 = map (reduce (+) 0.0f32) histC'
    let hist1 = map (reduce (+) 0.0f32) <| transpose histC' 

    in  (length hist1, length hist2, hist1, hist2, histC)

entry mkImgRegDeriv [n][h] (dataR: [n]f32) (vals: [n]f32) (hist_bar: [h]f32) : ([n]f32, [n]f32) =
    let lb1 = reduce_comm f32.min f32.highest dataR
    let ub1 = reduce_comm f32.max f32.lowest  dataR
    let lb2 = reduce_comm f32.min f32.highest vals
    let h1 = 4 + i32.f32 (ub1 - lb1)
    let det = replicate n 1.0f32
    -- this should not actually be recomputed but saved from the histogram
    -- computation and passed as parameters from where is called from pythorch

    let (dataR_bar, vals_bar) = unzip2 <|
        map3(\vr v d ->
                let t  = vr - (f32.floor vr)
                let t2 = t  * t
                let t3 = t2 * t
                let vals1 = [ (-t3+3.0*t2-3.0*t+1.0)/6.0, (3.0*t3-6.0*t2+4.0)/6.0
                            , (-3.0*t3+3.0*t2+3.0*t+1.0)/6.0,  t3/6.0 ]
                let idxR = 1 + i32.f32 (vr - lb1)
                let inds1  = [ idxR-1, idxR, idxR+1, idxR+2 ]
                
                let t  = v - (f32.floor v)
                let t2 = t  * t
                let t3 = t2 * t
                let vals2 = [ (-t3+3.0*t2-3.0*t+1.0)/6.0, (3.0*t3-6.0*t2+4.0)/6.0
                            , (-3.0*t3+3.0*t2+3.0*t+1.0)/6.0,  t3/6.0 ]
                let idx = 1 + i32.f32 (v - lb2)
                let inds2 = [ idx-1, idx, idx+1, idx+2 ]

                let vals1_bar = replicate 4i32 0.0f32
                let vals2_bar = replicate 4i32 0.0f32
                let (vals1_bar, vals2_bar) =
                    loop (vals1_bar, vals2_bar) for j < 4 do
                        let i2 = inds2[j]
                        let v2 = vals2[j]
                        let v2_bar = 0.0f32
                        let (vals1_bar, v2_bar) =
                            loop (vals1_bar, v2_bar) for k < 4 do
                                let i1 = inds1[k]
                                let v1 = vals1[k]
                                let v1_bar = 0.0f32
                                -- histogram computation was:
                                -- let v = d*v2*v1
                                -- let histC[h1*i2 + i1] += v
                                let v_bar  = hist_bar[h1*i2 + i1]
                                let v2_bar = v2_bar + v_bar * (d * v1)
                                let v1_bar = v1_bar + v_bar * (d * v2)
                                let vals1_bar[k] = vals1_bar[k] + v1_bar
                                in (vals1_bar, v2_bar)
                        let vals2_bar[j] = v2_bar
                        in  (vals1_bar, vals2_bar)

                let t  = v - (f32.floor v)
                let t_bar = 0
                let t_bar = t_bar + vals2_bar[0] * (-3.0*t2 + 6.0*t - 3.0) / 6.0
                let t_bar = t_bar + vals2_bar[1] * ( 3.0*t2 - 6.0*t) / 6.0
                let t_bar = t_bar + vals2_bar[2] * (-9.0*t2 + 6.0*t + 3.0) / 6.0
                let t_bar = t_bar + vals2_bar[3] * (3.0 * t2) / 6.0
                let v_bar = t_bar * 1.0

                let t  = vr - (f32.floor vr)
                let t_bar = 0
                let t_bar = t_bar + vals1_bar[0] * (-3.0*t2 + 6.0*t - 3.0) / 6.0
                let t_bar = t_bar + vals1_bar[1] * ( 3.0*t2 - 6.0*t) / 6.0
                let t_bar = t_bar + vals1_bar[2] * (-9.0*t2 + 6.0*t + 3.0) / 6.0
                let t_bar = t_bar + vals1_bar[3] * (3.0 * t2) / 6.0
                let vr_bar = t_bar * 1.0

                in (vr_bar, v_bar)

            ) dataR vals det
    in (dataR_bar, vals_bar)
