let reduce_by_index_rf 'a [m] [n] (rf: i32) (dest : *[m]a) (f : a -> a -> a) (ne : a) (is : [n]i32) (as : [n]a) : *[m]a =
  intrinsics.hist (rf, dest, f, ne, is, as) :> *[m]a

--for(int i=0;i<N;i++){
--    detSum+=det[i];
--    int idxR=(int)floor((dataR[i]-range[2]));
--    double t=(dataR[i])-floor(dataR[i]);
--    double t2=t*t;
--    double t3=t2*t;
--                
--    double tr_val[4]={-t3+3*t2-3*t+1, 3*t3-6*t2+4, -3*t3+3*t2+3*t+1,  t3};
--    for (int nn=0;nn<4;nn++){
--        pR[idxR+nn-1]+=det[i]*tr_val[nn]/6;
--    }
--    t=val[i]-(int)(val[i]);
--    //mexPrintf("%f %f\n",vals[i],dataR[i]);
--    t2=t*t;
--    t3=t2*t;
--                
--    double t_val[4]={-t3+3*t2-3*t+1, 3*t3-6*t2+4, -3*t3+3*t2+3*t+1,  t3};
--    int dd=(int)((val[i]));
--    for(int m=0;m<4;m++)
--    {
--        pI[dd+m-1]+=det[i]*t_val[m]/6;
--        for (int nn=0;nn<4;nn++){
--            pRI[idxR+nn-1+(int)no_bins[1]*(dd+m-1)]+=det[i]*t_val[m]*tr_val[nn]/36;
--        }
--    }
--}

let mkImgRegHisto [n] (dataR: [n]f32) (vals: [n]f32) : ([]f32, []f32, []f32) =
    let lb1 = reduce_comm f32.min f32.highest dataR
    let ub1 = reduce_comm f32.max f32.lowest  dataR

    let lb2 = reduce_comm f32.min f32.highest vals
    let ub2 = reduce_comm f32.max f32.lowest  vals

    let h1 = 4 + i32.f32 (ub1 - lb1)
    let h2 = 4 + i32.f32 (ub2 - lb2)

    let det = replicate n 1.0f32

    let (h1_inp, h2_inp, hc_inp) = unzip3 <|
        map3(\vr v d : ( [4](i32,f32), [4](i32,f32), [16](i32,f32) ) ->
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

                let resC = ( flatten <|
                    map2(\ i2 v2 ->
                            map2(\i1 v1 -> ( h1*i2 + i1,  d * v2 * v1 )
                                ) inds1 vals1
                        ) inds2 vals2 
                    ) :> [16](i32,f32)

                let vals1' = map (*d) vals1
                let vals2' = map (*d) vals2
                in  (zip inds1 vals1', zip inds2 vals2', resC)
            ) dataR vals det --(iota n)
    let (h1_inds, h1_vals) = unzip <| flatten h1_inp
    let (h2_inds, h2_vals) = unzip <| flatten h2_inp
    let (hc_inds, hc_vals) = unzip <| flatten hc_inp
    let hist1 = reduce_by_index_rf 1i32 (replicate h1 0.0f32) (+) 0.0f32 h1_inds h1_vals
    let hist2 = reduce_by_index_rf 1i32 (replicate h2 0.0f32) (+) 0.0f32 h2_inds h2_vals
    let hc = h1 * h2
    let histC = reduce_by_index_rf 1i32 (replicate hc 0.0f32) (+) 0.0f32 hc_inds hc_vals
    in  (hist1, hist2, histC)

let mkImgRegDeriv [n][h] (dataR: [n]f32) (vals: [n]f32) (hist_bar: [h]f32) : ([n]f32, [n]f32) =
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


let main [n] (dataR: [n]f32) (vals: [n]f32) : (i32, i32, []f32, []f32, []f32) =
    let (hist1, hist2, histC) = mkImgRegHisto dataR vals
    in  (length hist1, length hist2, hist1, hist2, histC)
--let main (v: f32) : i32 = i32.f32 v