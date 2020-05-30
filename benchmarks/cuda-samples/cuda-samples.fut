-- ==
-- entry: words
-- random input { [64]i32 63 [16777216]i32 }
-- random input { [256]i32 255 [16777216]i32 }

entry words [n][H] (dest: *[H]i32) (mask: i32) (xs: [n]i32) =
  let index i = #[unsafe] ((xs[i//4] >> (8*(i%%4)))&mask)
  in reduce_by_index dest (+) 0
                     (map index (iota (n*4)))
                     (replicate (n*4) 1)

-- ==
-- entry: bytes
-- random input { [64]i32 63 [67108864]u8 }
-- random input { [256]i32 255 [67108864]u8 }

entry bytes [n][H] (dest: *[H]i32) (mask: i32) (xs: [n]u8) =
  reduce_by_index dest (+) 0
                  (map (i32.u8 >-> (&mask)) xs)
                  (replicate n 1)
