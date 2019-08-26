-- ==
-- entry: points_in_clusters cluster_sums both_fused
-- compiled random input { 5 [500000][2]f32 [500000]i32 }
-- compiled random input { 5 [500000][35]f32 [500000]i32 }

entry points_in_clusters [n][d] (k: i32) (points: [n][d]f32) (membership: [n]i32) =
  reduce_by_index (replicate k 0) (+) 0
                  (map (%k) membership) (replicate n 1)

entry cluster_sums [n][d] (k: i32) (points: [n][d]f32) (membership: [n]i32) =
  reduce_by_index (replicate k (replicate d 0)) (map2 (+)) (replicate d 0)
                  (map (%k) membership)
                  points

entry both_fused [n][d] (k: i32) (points: [n][d]f32) (membership: [n]i32)=
  (points_in_clusters k points membership,
   cluster_sums k points membership)
