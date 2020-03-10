-- ==
-- random input { 64 63 [67108864]u8 }
-- random input { 256 255 [67108864]u8 }

let main [n] (H: i32) (mask: i32) (xs: [n]u8) =
  reduce_by_index (replicate H 0) (+) 0i32
                  (map (i32.u8 >-> (&mask)) xs)
                  (replicate n 1)
