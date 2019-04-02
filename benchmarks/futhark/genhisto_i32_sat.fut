-- ==
-- random input { 10 [10000000]i32 } auto output
-- random input { 100 [10000000]i32 } auto output
-- random input { 1000 [10000000]i32 } auto output
-- random input { 10000 [10000000]i32 } auto output
-- random input { 100000 [10000000]i32 } auto output
-- random input { 1000000 [10000000]i32 } auto output

let sat_add_u8 (x: i32) (y: i32): i32 =
  if x + y > i32.u8 u8.highest then i32.u8 u8.highest else x + y

let main [n] (k: i32) (is: [n]i32) =
  reduce_by_index (replicate k 0) sat_add_u8 0 (map (%k) is) (replicate n 1)
