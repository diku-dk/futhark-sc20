let main [n][m][p][q][t](jindex: [n]i32)
                        (iinr  : [m]i32)
                        (jjnr  : [p]i32)
                        (shift : [q]i32)
                        (types : [t]i32) =
  let max_i     = reduce_comm i32.max i32.lowest iinr
  let max_j     = reduce_comm i32.max i32.lowest jjnr
  let max_shift = reduce_comm i32.max i32.lowest shift
  let max_types = reduce_comm i32.max i32.lowest types
  let max_jindex= jindex[n-1]

  let rf1_ones = scatter (replicate t 0i32) (jjnr[:t]) (replicate t 1)
  let rf1 = t / reduce_comm (+) 0i32 rf1_ones

  let rf2_ones = scatter (replicate t 0i32) (jjnr[t:2*t] :> [t]i32) (replicate t 1)
  let rf2 = t / reduce_comm (+) 0i32 rf2_ones

  --in  (n,m,p,q,t, max_i, max_j, max_shift, max_types, max_jindex, rf1, rf2)
  in (m, p, jindex[m])
