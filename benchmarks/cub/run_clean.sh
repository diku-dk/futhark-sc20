#!/bin/bash

rm results_cub.txt

#H=31	& H=127	& H=505	& H=2041	& H=6141	& H=12281	& H=24569	
#H=49145 & H=196607	& H=393215	& H=786431	& H=1572863

./genhisto_i32 50000000 31   >> results_cub.txt
./genhisto_i32 50000000 127  >> results_cub.txt
./genhisto_i32 50000000 505  >> results_cub.txt
./genhisto_i32 50000000 2041 >> results_cub.txt
./genhisto_i32 50000000 6141 >> results_cub.txt
./genhisto_i32 50000000 12281 >> results_cub.txt
./genhisto_i32 50000000 24569 >> results_cub.txt
./genhisto_i32 50000000 49145 >> results_cub.txt
./genhisto_i32 50000000 196607 >> results_cub.txt
./genhisto_i32 50000000 393215 >> results_cub.txt
./genhisto_i32 50000000 786431 >> results_cub.txt
./genhisto_i32 50000000 1572863 >> results_cub.txt

./genhisto_i32_sat 50000000 31   >> results_cub.txt
./genhisto_i32_sat 50000000 127  >> results_cub.txt
./genhisto_i32_sat 50000000 505  >> results_cub.txt
./genhisto_i32_sat 50000000 2041 >> results_cub.txt
./genhisto_i32_sat 50000000 6141 >> results_cub.txt
./genhisto_i32_sat 50000000 12281 >> results_cub.txt
./genhisto_i32_sat 50000000 24569 >> results_cub.txt
./genhisto_i32_sat 50000000 49145 >> results_cub.txt
./genhisto_i32_sat 50000000 196607 >> results_cub.txt
./genhisto_i32_sat 50000000 393215 >> results_cub.txt
./genhisto_i32_sat 50000000 786431 >> results_cub.txt
./genhisto_i32_sat 50000000 1572863 >> results_cub.txt

./genhisto_argmin 50000000 31   >> results_cub.txt
./genhisto_argmin 50000000 127  >> results_cub.txt
./genhisto_argmin 50000000 505  >> results_cub.txt
./genhisto_argmin 50000000 2041 >> results_cub.txt
./genhisto_argmin 50000000 6141 >> results_cub.txt
./genhisto_argmin 50000000 12281 >> results_cub.txt
./genhisto_argmin 50000000 24569 >> results_cub.txt
./genhisto_argmin 50000000 49145 >> results_cub.txt
./genhisto_argmin 50000000 196607 >> results_cub.txt
./genhisto_argmin 50000000 393215 >> results_cub.txt
./genhisto_argmin 50000000 786431 >> results_cub.txt
./genhisto_argmin 50000000 1572863 >> results_cub.txt
