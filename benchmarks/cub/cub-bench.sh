#!/bin/sh
#
# A quick shell script for running the CUB implementations on their
# datasets.  Produces a CSV file on stdout.

set -e

IMPL=$1

Hs="31 127 505 2041 6141 12281 24569 49145 196607 393215 786431 1572863"
N=50000000

for H in $Hs; do
    f=$(mktemp)
    $IMPL $N $H $f >&2
    echo -n "$H, "
    cat $f
    echo
    rm -f $f
done
