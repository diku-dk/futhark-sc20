let reduce_by_index_rf 'a [m] [n] (rf: i32) (dest : *[m]a) (f : a -> a -> a) (ne : a) (is : [n]i32) (as : [n]a) : *[m]a =
  intrinsics.hist (rf, dest, f, ne, is, as)

-- ==
-- entry: hwd cas
-- compiled random input { 25 1 [50000000]i32 } auto output
-- compiled random input { 121 1 [50000000]i32 } auto output
-- compiled random input { 505 1 [50000000]i32 } auto output
-- compiled random input { 2041 1 [50000000]i32 } auto output
-- compiled random input { 6143 1 [50000000]i32 } auto output
-- compiled random input { 12287 1 [50000000]i32 } auto output
-- compiled random input { 24575 1 [50000000]i32 } auto output
-- compiled random input { 49151 1 [50000000]i32 } auto output
-- compiled random input { 196607 1 [50000000]i32 } auto output
-- compiled random input { 393215 1 [50000000]i32 } auto output
-- compiled random input { 786431 1 [50000000]i32 } auto output
-- compiled random input { 1572863 1 [50000000]i32 } auto output
--
-- compiled random input { 25 64 [50000000]i32 } auto output
-- compiled random input { 121 64 [50000000]i32 } auto output
-- compiled random input { 505 64 [50000000]i32 } auto output
-- compiled random input { 2041 64 [50000000]i32 } auto output
-- compiled random input { 6143 64 [50000000]i32 } auto output
-- compiled random input { 12287 64 [50000000]i32 } auto output
-- compiled random input { 24575 64 [50000000]i32 } auto output
-- compiled random input { 49151 64 [50000000]i32 } auto output
-- compiled random input { 196607 64 [50000000]i32 } auto output
-- compiled random input { 393215 64 [50000000]i32 } auto output
-- compiled random input { 786431 64 [50000000]i32 } auto output
-- compiled random input { 1572863 64 [50000000]i32 } auto output

let stride : i32 = 64--16

let index (H: i32) (RF: i32) (elm: i32) =
  i32.u32 (u32.i32 elm %% u32.max 1 (u32.i32 (H/RF))) * RF

entry hwd [n] (H: i32) (RF: i32) (vs: [n]i32) =
  reduce_by_index_rf RF (replicate H 0) (+) 0 (map (index H RF) vs) vs

let sat_add_u24 (x: i32) (y: i32): i32 =
  let sat_val = (1 << 24) - 1
  in if sat_val - x < y
     then sat_val else x + y

entry cas [n] (H: i32) (RF: i32) (vs: [n]i32) =
  reduce_by_index_rf RF (replicate H 0) sat_add_u24 0 (map (index H RF) vs) (map (%%4) vs)

-- ==
-- entry: xcg
--
-- compiled random input { 25 1 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 121 1 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 505 1 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 2041 1 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 6143 1 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 12287 1 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 24575 1 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 49151 1 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 196607 1 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 393215 1 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 786431 1 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 1572863 1 [50000000]i32 [50000000]i32 } auto output
--
-- compiled random input { 25 64 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 121 64 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 505 64 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 2041 64 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 6143 64 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 12287 64 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 24575 64 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 49151 64 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 196607 64 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 393215 64 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 786431 64 [50000000]i32 [50000000]i32 } auto output
-- compiled random input { 1572863 64 [50000000]i32 [50000000]i32 } auto output

let unpack (a: u64) : (i32, i32) =
  (i32.u64 a, i32.u64 (a >> 32))

let pack (i: i32, x: i32) : u64 =
  u64.i32 i | (u64.i32 x << 32)

let argmax (a: u64) (b: u64): u64 =
  let (i, x) = unpack a
  let (j, y) = unpack b
  in pack (if x < y then (i, x)
           else if y < x then (j, y)
           else if i < j then (i, x)
           else (j, y))

entry xcg [n] (H: i32) (RF: i32) (vs_a: [n]i32) (vs_b: [n]i32) =
  reduce_by_index_rf RF
                     (replicate H (pack (i32.highest, i32.lowest)))
                     argmax (pack (i32.highest, i32.lowest))
                     (map (index H RF) vs_a) (map pack (zip vs_a vs_b))
