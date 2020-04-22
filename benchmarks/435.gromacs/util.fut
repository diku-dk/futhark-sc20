type real   =  f32
let nul     =  0.0f32
let one     =  1.0f32
let six     =  6.0f32
let twelve  = 12.0f32


let sgmscan 't [n] (op: (t -> t -> t)) (ne: t) (flag: [n]i8) (vals: [n]t) : [n]t =
  let (_, res) = scan (\ (f1,v1) (f2,v2) ->
                            let f = f1 | f2
                            let v = if f2 != 0 then v2 else op v1 v2
                            in  (f,v)
                      ) (0, ne) (zip flag vals) |> unzip
  in  res

let reduce_by_index_rf 'a [m] [n] (rf: i32) (dest : *[m]a) (f : a -> a -> a) (ne : a) (is : [n]i32) (as : [n]a) : *[m]a =
  intrinsics.hist (rf, dest, f, ne, is, as) :> *[m]a


let binSearch [N] (gid: i32) (A: [N]i32) : i32 =
 (  loop (_,L,R) = (-1i32, 0i32, N-1i32)
    while L <= R do
        let m = (L + R) / 2i32 in
        if (A[m] <= gid)
        then if (m < N-1i32) && (A[m+1] > gid)
             then (m, 1, 0) -- return m
             else (-1i32, m + 1, R) 
        else -- A[m] > gid
             if (m > 0) && (A[m-1] <= gid)
             then (m-1, 1, 0) -- return m-1
             else (-1i32, L, m - 1)
 ).0

let calcRF [h] (inds: [h]i32) =
    let RF0 = scatter (replicate h 0i32) inds (replicate h 1i32)
           |> reduce (+) 0
    in  3*(h/RF0)
