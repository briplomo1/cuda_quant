#!/usr/bin/env bash
# Every Algorithm 1 experiment for the report, run on the GPU server from the project folder:
#   bash scripts/run_experiments.sh          # all six
#   bash scripts/run_experiments.sh 12345    # only the experiments listed
# Needs data/ from "prepare.py --images 4096" and a Release build in cmake-build-release.
# Logs go to results/raw; scripts/summarize.py turns them into tables and plots.
# Runs one after another, so no experiment's timing is disturbed by another.
set -euo pipefail
cd "$(dirname "$0")/.."
B=cmake-build-release
OUT=results/raw
mkdir -p "$OUT"
LAYERS=$(seq -w 1 19) # layer 00's input is the image, so it isn't calibrated
unset DISPLAY          # otherwise Open MPI probes for an X display over ssh and prints warnings
RUN=${1:-123456}       # which experiments to run
want() { [[ $RUN == *$1* ]]; }

# P2 on p ranks: one per physical core up to 22, then hardware threads (the CPU has 22
# cores, 44 threads)
mpi() {
    local p=$1
    shift
    if [ "$p" -le 22 ]; then
        mpirun -np "$p" --bind-to core --map-by core "$B/calib_mpi" "$@"
    else
        mpirun -np "$p" --use-hwthread-cpus --bind-to hwthread "$B/calib_mpi" "$@"
    fi
}

if want 1; then
    echo "1/6 P1 on every layer"
    # Thresholds, checks against the CPU, GPU and CPU time per step, histogram variants
    for L in $LAYERS; do
        echo "== layer $L"
        $B/calib "data/layer${L}_act.bin" --images 512
    done > "$OUT/p1_layers.txt"
fi

if want 2; then
    echo "2/6 P1 vs number of blocks"
    # absmax and histogram with 1 .. 660 blocks; the searches always use one block per candidate
    for G in 1 2 4 8 16 32 64 128 256 512 660; do
        echo "== grid $G"
        $B/calib data/layer01_act.bin --images 512 --grid "$G" --no-cpu
    done > "$OUT/p1_grid.txt"
fi

if want 3; then
    echo "3/6 P1 vs problem size"
    for N in 512 1024 2048 4096; do
        echo "== images $N"
        $B/calib data/layer01_act.bin --images "$N"
    done > "$OUT/p1_size.txt"
fi

if want 4; then
    echo "4/6 P2 strong scaling"
    # Fixed work, more ranks: one layer of each size at 512 images, and layer 01 at 4096
    for CASE in 01:512 08:512 14:512 19:512 01:4096; do
        L=${CASE%:*}
        N=${CASE#*:}
        for P in 1 2 4 8 16 22 32 44; do
            echo "== layer $L images $N p $P"
            mpi "$P" "data/layer${L}_act.bin" --images "$N"
        done
    done > "$OUT/p2_strong.txt"
fi

if want 5; then
    echo "5/6 P2 weak scaling"
    # 64 images per rank on layer 01
    for P in 1 2 4 8 16 22 32 44; do
        echo "== p $P"
        mpi "$P" data/layer01_act.bin --images $((64 * P))
    done > "$OUT/p2_weak.txt"
fi

if want 6; then
    echo "6/6 smoothing"
    # Radius 0 (off), 2, 4 and 8 (sigma 1, 2, 4). The spread of T uses 8 disjoint subsets of
    # each size on one layer per shape; every other layer gets only the first 512 images, which
    # the accuracy runs need.
    SPREAD_LAYERS="01 08 14 19"
    smooth_run() { # layer size subset radius
        local line
        line=$($B/calib "data/layer$1_act.bin" --first $(($3 * $2)) --images "$2" --radius "$4" --no-cpu | grep '^result:')
        echo "layer $1 size $2 subset $3 $line"
    }
    for L in $LAYERS; do
        for R in 0 2 4 8; do
            if [[ " $SPREAD_LAYERS " == *" $L "* ]]; then
                for S in 32 64 128 512; do
                    for K in 0 1 2 3 4 5 6 7; do smooth_run "$L" "$S" "$K" "$R"; done
                done
            else
                smooth_run "$L" 512 0 "$R"
            fi
        done
    done > "$OUT/smoothing.txt"
fi

echo "done: $OUT"
