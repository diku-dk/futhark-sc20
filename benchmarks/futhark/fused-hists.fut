let reduce_by_index_rf 'a [m] [n] (rf: i32) (dest : *[m]a) (f : a -> a -> a) (ne : a) (is : [n]i32) (as : [n]a) : *[m]a =
  intrinsics.hist (rf, dest, f, ne, is, as) :> *[m]a

let index1 (m:i32) (elm: i32) = ((elm >>  0) & m) 
let index2 (m:i32) (elm: i32) = ((elm >>  8) & m) -- << 3
let index3 (m:i32) (elm: i32) = ((elm >> 16) & m) -- 255 = 0XFF
let index4 (m:i32) (elm: i32) = ((elm >> 24) & m) 

let index (m:i32) (vs: []i32) (ind: i32) =
  let elm = unsafe vs[ind >> 2] 
  let r   = ind & 3
  let shf = r * 8
  in  ((elm >> shf) & m)


-- ==
-- entry: test_image_8b_4C_H64 test_image_8b_4C_H256 test_image_8b_1C_H64 test_image_8b_1C_H256 
-- compiled input @ real-images/logo_small.raw.in

-- compiled input @ real-images/feli.raw.in
-- compiled input @ real-images/tex_h.raw.in

let test_image_8b_4C [n] (H: i32) (vs: [n]i32) =
  let RF    = 1
  let m     = H-2
  let ones  = replicate n 1i32 in
  let hist1 = reduce_by_index_rf RF (replicate H 0) (+) 0 (map (index1 m) vs) ones --|> intrinsics.opaque
  let hist2 = reduce_by_index_rf RF (replicate H 0) (+) 0 (map (index2 m) vs) ones --|> intrinsics.opaque
  let hist3 = reduce_by_index_rf RF (replicate H 0) (+) 0 (map (index3 m) vs) ones --|> intrinsics.opaque
  let hist4 = reduce_by_index_rf RF (replicate H 0) (+) 0 (map (index4 m) vs) ones --|> intrinsics.opaque
  in  (hist1, hist2, hist3, hist4)

entry test_image_8b_4C_H256 [n] (vs: [n]i32) =
  test_image_8b_4C 257 vs

entry test_image_8b_4C_H64  [n] (vs: [n]i32) =
  test_image_8b_4C 65 vs


entry test_image_8b_1C [n] (H: i32) (vs: [n]i32) =
  let RF= 1
  let m = H-2
  in  reduce_by_index_rf RF (replicate H 0) (+) 0 
                         (map (index m vs) (iota (4*n)))
                         (replicate (4*n) 1i32)

entry test_image_8b_1C_H256 [n] (vs: [n]i32) =
  test_image_8b_1C 257 vs

entry test_image_8b_1C_H64  [n] (vs: [n]i32) =
  test_image_8b_1C 65 vs

-- futhark bench --backend=cuda fused-hists.fut --pass-option=--nvrtc-option=-arch=compute_35 --pass-option=--default-num-groups=36 --pass-option=--default-group-size=1024 -r 1000
