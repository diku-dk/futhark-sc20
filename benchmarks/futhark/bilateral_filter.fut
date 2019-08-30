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
                    (map ((*256) >-> t32) cell)
    |> map (r32 >-> (/256))
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

let lerp (v0, v1, t) =
  v0 + (v1-v0)*f32.max 0 (f32.min 1 t)

let shape_3d [n][m][k] 't (_: [n][m][k]t) = (n, m, k)

entry bilateral_filter [n][m] (s_sigma: i32) (r_sigma: f32) (I: [n][m]f32) =
  let grid = bilateral_grid (r32 s_sigma) r_sigma I
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

  let (_, _, k) = shape_3d blury

  let sample arr f x y =
    let xf = r32 (x % s_sigma) / r32 s_sigma
    let yf = r32 (y % s_sigma) / r32 s_sigma
    let xi = x / s_sigma
    let yi = y / s_sigma
    let zv = unsafe I[x,y] * (1/r_sigma)
    let zi = t32 zv
    let zf = zv - r32 zi
    let pick (i, j, l) = unsafe (f arr[i32.min (n-1) (i32.max 0 i),
                                       i32.min (m-1) (i32.max 0 j),
                                       i32.min (k-1) (i32.max 0 l)])
    in lerp(lerp(lerp(pick(xi, yi, zi), pick(xi+1, yi, zi), xf),
                 lerp(pick(xi, yi+1, zi), pick(xi+1, yi+1, zi), xf), yf),
            lerp(lerp(pick(xi, yi, zi+1), pick(xi+1, yi, zi+1), xf),
                 lerp(pick(xi, yi+1, zi+1), pick(xi+1, yi+1, zi+1), xf), yf), zf)


  in map2 (map2 (/))
          (tabulate_2d n m (sample blury (.1)))
          (tabulate_2d n m (sample blury ((.2) >-> r32)))
