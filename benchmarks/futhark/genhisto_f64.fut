-- ==
-- random input { 10 [10000000]i32 [10000000]f64 } auto output
-- random input { 100 [10000000]i32 [10000000]f64 } auto output
-- random input { 1000 [10000000]i32 [10000000]f64  } auto output
-- random input { 10000 [10000000]i32 [10000000]f64 } auto output
-- random input { 100000 [10000000]i32 [10000000]f64 } auto output
-- random input { 1000000 [10000000]i32 [10000000]f64 } auto output

let main [n] (k: i32) (is: [n]i32) (xs: [n]f64) =
  reduce_by_index (replicate k 0) (+) 0 (map (%k) is) xs
