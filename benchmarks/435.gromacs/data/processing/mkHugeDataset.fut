
let equal8 [np] (a1: [np]i32) (a2: [np]i32) (a3: [np]i32) (a4: [np]i32)
                (a5: [np]i32) (a6: [np]i32) (a7: [np]i32) (a8: [np]i32) : bool =
    map (\ i -> let (a,b,c,d,e,f,g,h) = (a1[i], a2[i], a3[i], a4[i], a5[i], a6[i], a7[i], a8[i]) 
                 in a==b && b==c && c==d && d==e && e==f && f==g && g==h
        ) (iota np) |> reduce_comm (&&) true

let equal_cols [q][np] (a2d: [q][np]i32) =
    transpose a2d |> map (\col -> let el = col[0] in
                                  map (==el) col |> reduce_comm (&&) true
                         )
                  |> reduce_comm (&&) true

let shape1 [n] (jindex: [n]i32) : []i32 =
    map (\i -> jindex[i+1] - jindex[i]) (iota (n-1))

let shape2 [n1][n2] (jindex1: [n1]i32) (jindex2: [n2]i32) : []i32 =
    (shape1 jindex1) ++ (shape1 jindex2)

let shape4 [n1][n2][n3][n4] (jindex1: [n1]i32) (jindex2: [n2]i32)
                            (jindex3: [n3]i32) (jindex4: [n4]i32) : []i32 =
    (shape2 jindex1 jindex2) ++ (shape2 jindex3 jindex4)


let shape8 (jindex1: []i32) (jindex2: []i32) (jindex3: []i32) (jindex4: []i32) 
           (jindex5: []i32) (jindex6: []i32) (jindex7: []i32) (jindex8: []i32) : []i32 =
    (shape4 jindex1 jindex2 jindex3 jindex4) ++ (shape4 jindex5 jindex6 jindex7 jindex8)


-- jindex_i : [ni+1]i32
-- iinr_i   : [ni]  i32
-- jjnr_i   : [mi]  i32
-- shift_i  : [ni]  i32
-- types_i  : [np]  i32  == 23178 particles  

let main [np] [n1][m1] [n2][m2] [n3][m3] [n4][m4] [n5][m5] [n6][m6] [n7][m7] 
              [n8][m8] [n9][m9] [n10][m10] [n11][m11] [n12][m12] [n13][m13]
              [n14][m14] [n15][m15] [n16][m16] [n17][m17] [n18][m18] [n19][m19]
              [n20][m20] [n21][m21] [n22][m22] 
        (jindex1: []i32) (iinr1: [n1]i32) (jjnr1: [m1]i32) (shift1: [n1]i32) (types1: [np]i32)
        (jindex2: []i32) (iinr2: [n2]i32) (jjnr2: [m2]i32) (shift2: [n2]i32) (types2: [np]i32)
        (jindex3: []i32) (iinr3: [n3]i32) (jjnr3: [m3]i32) (shift3: [n3]i32) (types3: [np]i32)
        (jindex4: []i32) (iinr4: [n4]i32) (jjnr4: [m4]i32) (shift4: [n4]i32) (types4: [np]i32)
        (jindex5: []i32) (iinr5: [n5]i32) (jjnr5: [m5]i32) (shift5: [n5]i32) (types5: [np]i32)
        (jindex6: []i32) (iinr6: [n6]i32) (jjnr6: [m6]i32) (shift6: [n6]i32) (types6: [np]i32)
        (jindex7: []i32) (iinr7: [n7]i32) (jjnr7: [m7]i32) (shift7: [n7]i32) (types7: [np]i32)
        (jindex8: []i32) (iinr8: [n8]i32) (jjnr8: [m8]i32) (shift8: [n8]i32) (types8: [np]i32)
        (jindex9: []i32) (iinr9: [n9]i32) (jjnr9: [m9]i32) (shift9: [n9]i32) (types9: [np]i32)
        (jindex10: []i32) (iinr10: [n10]i32) (jjnr10: [m10]i32) (shift10: [n10]i32) (types10: [np]i32)
        (jindex11: []i32) (iinr11: [n11]i32) (jjnr11: [m11]i32) (shift11: [n11]i32) (types11: [np]i32)
        (jindex12: []i32) (iinr12: [n12]i32) (jjnr12: [m12]i32) (shift12: [n12]i32) (types12: [np]i32)
        (jindex13: []i32) (iinr13: [n13]i32) (jjnr13: [m13]i32) (shift13: [n13]i32) (types13: [np]i32)
        (jindex14: []i32) (iinr14: [n14]i32) (jjnr14: [m14]i32) (shift14: [n14]i32) (types14: [np]i32)
        (jindex15: []i32) (iinr15: [n15]i32) (jjnr15: [m15]i32) (shift15: [n15]i32) (types15: [np]i32)
        (jindex16: []i32) (iinr16: [n16]i32) (jjnr16: [m16]i32) (shift16: [n16]i32) (types16: [np]i32)
        (jindex17: []i32) (iinr17: [n17]i32) (jjnr17: [m17]i32) (shift17: [n17]i32) (types17: [np]i32)
        (jindex18: []i32) (iinr18: [n18]i32) (jjnr18: [m18]i32) (shift18: [n18]i32) (types18: [np]i32)
        (jindex19: []i32) (iinr19: [n19]i32) (jjnr19: [m19]i32) (shift19: [n19]i32) (types19: [np]i32)
        (jindex20: []i32) (iinr20: [n20]i32) (jjnr20: [m20]i32) (shift20: [n20]i32) (types20: [np]i32)
        (jindex21: []i32) (iinr21: [n21]i32) (jjnr21: [m21]i32) (shift21: [n21]i32) (types21: [np]i32)
        (jindex22: []i32) (iinr22: [n22]i32) (jjnr22: [m22]i32) (shift22: [n22]i32) (types22: [np]i32)
        --(jindex23: []i32) (iinr23: [n23]i32) (jjnr23: [m23]i32) (shift23: [n23]i32) (types23: [np]i32)
        --(jindex24: []i32) (iinr24: [n24]i32) (jjnr24: [m24]i32) (shift24: [n24]i32) (types24: [np]i32)
  =
    let types_equal =  equal_cols   [ types1, types2, types3, types4, types5, types6, types7, types8, types9 
                                    , types10, types11, types12, types13, types14, types15, types16
                                    , types17, types18, types19, types20, types21, types22]

    let (q1, q2, q3, q4, q5, q6, q7, q8) = (jindex1[n1], jindex2[n2], jindex3[n3], jindex4[n4],
                                            jindex5[n5], jindex6[n6], jindex7[n7], jindex8[n8])

    let (q9, q10, q11, q12, q13, q14, q15, q16) = 
        (jindex9[n9], jindex10[n10], jindex11[n11], jindex12[n12],
         jindex13[n13], jindex14[n14], jindex15[n15], jindex16[n16])

    let (q17, q18, q19, q20, q21, q22) = 
        (jindex17[n17], jindex18[n18], jindex19[n19], jindex20[n20], jindex21[n21], jindex22[n22])
    

    let iinr = iinr1 ++ iinr2 ++ iinr3 ++ iinr4 ++ iinr5 ++ iinr6 ++ iinr7 ++ iinr8 ++ iinr9
               ++ iinr10 ++ iinr11 ++ iinr12 ++ iinr13 ++ iinr14 ++ iinr15 ++ iinr16
               ++ iinr17 ++ iinr18 ++ iinr19 ++ iinr20 ++ iinr21 ++ iinr22

    let jjnr = jjnr1[:q1] ++ jjnr2[:q2] ++ jjnr3[:q3] ++ jjnr4[:q4] ++ jjnr5[:q5] ++ jjnr6[:q6]
            ++ jjnr7[:q7] ++ jjnr8[:q8] ++ jjnr9[:q9] ++ jjnr10[:q10] ++ jjnr11[:q11] ++ jjnr12[:q12]
            ++ jjnr13[:q13] ++ jjnr14[:q14] ++ jjnr15[:q15] ++ jjnr16[:q16] ++ jjnr17[:q17]
            ++ jjnr18[:q18] ++ jjnr19[:q19] ++ jjnr20[:q20] ++ jjnr21[:q21] ++ jjnr22[:q22]

    let shift= shift1 ++ shift2 ++ shift3 ++ shift4 ++ shift5 ++ shift6 ++ shift7 ++ shift8 ++ shift9
               ++ shift10 ++ shift11 ++ shift12 ++ shift13 ++ shift14 ++ shift15 ++ shift16
               ++ shift17 ++ shift18 ++ shift19 ++ shift20 ++ shift21 ++ shift22

    --let len_jindex = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 +
    --                 n13 + n14 + n15 + n16 + n17 + n18 + n19 + n20 + n21 + n22 + 1

    let shape = (shape8 jindex1 jindex2 jindex3 jindex4 jindex5 jindex6 jindex7 jindex8) ++
                (shape8 jindex9 jindex10 jindex11 jindex12 jindex13 jindex14 jindex15 jindex16) ++
                (shape4 jindex17 jindex18 jindex19 jindex20) ++ (shape2 jindex21 jindex22)

    let jindex = [0] ++ (scan (+) 0i32 shape)

    in  (jindex, iinr, jjnr, shift, types1)
                            

-- cat 'nri=1390_jindex=1390_iinr=1389_jjnr=62938_shift=1389_types=23178.in' 'nri=1390_jindex=1390_iinr=1389_jjnr=63066_shift=1389_types=23178.in' 'nri=1391_jindex=1391_iinr=1390_jjnr=61782_shift=1390_types=23178.in' 'nri=1392_jindex=1392_iinr=1391_jjnr=62746_shift=1391_types=23178.in' 'nri=1392_jindex=1392_iinr=1391_jjnr=63047_shift=1391_types=23178.in' 'nri=1392_jindex=1392_iinr=1391_jjnr=63113_shift=1391_types=23178.in' 'nri=1393_jindex=1393_iinr=1392_jjnr=62539_shift=1392_types=23178.in' 'nri=1393_jindex=1393_iinr=1392_jjnr=63169_shift=1392_types=23178.in' 'nri=1394_jindex=1394_iinr=1393_jjnr=62109_shift=1393_types=23178.in' 'nri=1394_jindex=1394_iinr=1393_jjnr=62827_shift=1393_types=23178.in' 'nri=1394_jindex=1394_iinr=1393_jjnr=62876_shift=1393_types=23178.in' 'nri=1395_jindex=1395_iinr=1394_jjnr=63003_shift=1394_types=23178.in' 'nri=1395_jindex=1395_iinr=1394_jjnr=63236_shift=1394_types=23178.in' 'nri=1396_jindex=1396_iinr=1395_jjnr=62034_shift=1395_types=23178.in' 'nri=1396_jindex=1396_iinr=1395_jjnr=62255_shift=1395_types=23178.in' 'nri=1396_jindex=1396_iinr=1395_jjnr=62854_shift=1395_types=23178.in' 'nri=1397_jindex=1397_iinr=1396_jjnr=62503_shift=1396_types=23178.in' 'nri=1397_jindex=1397_iinr=1396_jjnr=62952_shift=1396_types=23178.in' 'nri=1397_jindex=1397_iinr=1396_jjnr=62991_shift=1396_types=23178.in' 'nri=1397_jindex=1397_iinr=1396_jjnr=63448_shift=1396_types=23178.in' 'nri=1399_jindex=1399_iinr=1398_jjnr=62881_shift=1398_types=23178.in' 'nri=1399_jindex=1399_iinr=1398_jjnr=63100_shift=1398_types=23178.in' > huge-indarr.in
