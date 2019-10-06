-- ==
-- entry: hwd cas
-- compiled random input { 25 1 [50000000]i32 }
-- compiled random input { 121 1 [50000000]i32 }
-- compiled random input { 505 1 [50000000]i32 }
-- compiled random input { 2041 1 [50000000]i32 }
-- compiled random input { 6143 1 [50000000]i32 }
-- compiled random input { 12287 1 [50000000]i32 }
-- compiled random input { 24575 1 [50000000]i32 }
-- compiled random input { 49151 1 [50000000]i32 }
--
-- compiled random input { 25 64 [50000000]i32 }
-- compiled random input { 121 64 [50000000]i32 }
-- compiled random input { 505 64 [50000000]i32 }
-- compiled random input { 2041 64 [50000000]i32 }
-- compiled random input { 6143 64 [50000000]i32 }
-- compiled random input { 12287 64 [50000000]i32 }
-- compiled random input { 24575 64 [50000000]i32 }
-- compiled random input { 49151 64 [50000000]i32 }

let stride : i32 = 16

let index (H: i32) (RF: i32) (elm: i32) =
  (elm % i32.max 1 (H/RF))*RF

entry hwd [n] (H: i32) (RF: i32) (vs: [n]i32) =
  reduce_by_index (replicate H 0) (+) 0 (map (index H RF) (iota n)) vs

let sat_add_u24 (x: i32) (y: i32): i32 =
  let sat_val = (1 << 24) - 1
  in if x + y > sat_val
     then sat_val else x + y

entry cas [n] (H: i32) (RF: i32) (vs: [n]i32) =
  reduce_by_index (replicate H 0) sat_add_u24 0 (map (index H RF) (iota n)) vs

let argmax (i: i32, x: i32) (j: i32, y: i32): (i32, i32) =
  if x < y then (i, x)
  else if y < x then (j, y)
  else if i < j then (i, x)
  else (j, y)

-- ==
-- entry: xcg
--
-- compiled random input { 25 1 [50000000]i32 [50000000]i32 }
-- compiled random input { 121 1 [50000000]i32 [50000000]i32 }
-- compiled random input { 505 1 [50000000]i32 [50000000]i32 }
-- compiled random input { 2041 1 [50000000]i32 [50000000]i32 }
-- compiled random input { 6143 1 [50000000]i32 [50000000]i32 }
-- compiled random input { 12287 1 [50000000]i32 [50000000]i32 }
-- compiled random input { 24575 1 [50000000]i32 [50000000]i32 }
-- compiled random input { 49151 1 [50000000]i32 [50000000]i32 }
--
-- compiled random input { 25 64 [50000000]i32 [50000000]i32 }
-- compiled random input { 121 64 [50000000]i32 [50000000]i32 }
-- compiled random input { 505 64 [50000000]i32 [50000000]i32 }
-- compiled random input { 2041 64 [50000000]i32 [50000000]i32 }
-- compiled random input { 6143 64 [50000000]i32 [50000000]i32 }
-- compiled random input { 12287 64 [50000000]i32 [50000000]i32 }
-- compiled random input { 24575 64 [50000000]i32 [50000000]i32 }
-- compiled random input { 49151 64 [50000000]i32 [50000000]i32 }

entry xcg [n] (H: i32) (RF: i32) (vs_a: [n]i32) (vs_b: [n]i32) =
  reduce_by_index (replicate H (i32.highest, i32.lowest))
                  argmax (i32.highest, i32.lowest)
                  (map (index H RF) (iota n)) (zip vs_a vs_b)
  |> unzip
