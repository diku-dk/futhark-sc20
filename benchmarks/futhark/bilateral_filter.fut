-- ==
-- entry: bilateral_filter
-- compiled input @ gray.in

let tile [n][m] 'a (bn: i32) (bm: i32) (xss: [n][m]a): [][][]a =
  let tiles_per_n = n / bn
  let tiles_per_m = m / bm
  let mk_tile bx by =
    tabulate (bn*bm) (\i -> let x = bx * bn + (i / bn)
                            let y = by * bm + (i % bn)
                            in unsafe xss[x, y])
  in tabulate_2d tiles_per_n tiles_per_m mk_tile

let bilateral_grid [nx][ny] (s_sigma: f32) (r_sigma: f32) (I: [nx][ny]f32) : [][][](f32,i32) =
  let nz' = t32 (1/r_sigma + 0.5)
  let bin v = t32 (v/r_sigma + 0.5)
  let I_tiled = tile (t32 (f32.round s_sigma)) (t32 (f32.round s_sigma)) I
  let intensity cell =
    reduce_by_index (replicate nz' 0) (+) 0
                    (cell |> map bin)
                    cell
  let count cell =
    reduce_by_index (replicate nz' 0) (+) 0
                    (cell |> map bin)
                    (map (const 1) cell)
  in map2 (map2 zip)
          (map (map intensity) I_tiled)
          (map (map count) I_tiled)


let fivepoint [n] 'a (op: a -> a -> a) (scale: a -> f32 -> a) (xs: [n]a) =
  let pick i = unsafe xs[i32.min (n-1) (i32.max 0 i)]
  in tabulate n (\i -> pick (i-2) `op` scale (pick (i-1)) 4 `op`
                       scale (pick i) 6 `op`
                       scale (pick (i+1)) 4 `op` pick (i+2))

entry bilateral_filter s_sigma r_sigma I =
  let grid = bilateral_grid s_sigma r_sigma I
  let smoothen' = map (map (fivepoint (\(x1,y1) (x2,y2) -> (x1+x2, y1+y2))
                                      (\(x, y) c -> (x * c, y * t32 c))))
  let blurz = smoothen' grid
  let blurx = blurz |>
              transpose |>
              map transpose |>
              smoothen' |>
              map transpose |>
              transpose
  let blury = blurx |>
              map transpose |>
              smoothen' |>
              map transpose
  in blury
