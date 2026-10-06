#!/usr/bin/env bash
# One command to produce every benchmark number (guide Step 8):
# results/results.csv and the README table.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python

TOOLCHAIN="$(${CXX:-c++} --version | head -1 | sed 's/ version /-/; s/ (.*//; s/ /-/g')"
ENV="$(git rev-parse --short HEAD),$(uname -m),$(uname -s),${TOOLCHAIN},$(date +%F)"

$PY scripts/make_vectors.py --check
cmake -B build-bench -DCMAKE_BUILD_TYPE=Release -DKALMAN_BUILD_BENCH=ON -DKALMAN_BUILD_TESTS=OFF ${CMAKE_ARGS:-}
cmake --build build-bench

mkdir -p results
rm -f results/results.csv
echo "library,library_version,scenario,filter,precision,metric,value,unit,commit,cpu,os,toolchain,date" > results/results.csv

./build-bench/bench/bench --benchmark_repetitions=30 --benchmark_report_aggregates_only=true \
    --benchmark_out=results/gbench.json --benchmark_out_format=json
$PY scripts/gbench_to_csv.py results/gbench.json "$ENV"
./build-bench/bench/accuracy "$ENV" >> results/results.csv

if [ -x build-bench/bench/bench_alloc ]; then
    ./build-bench/bench/bench_alloc "$ENV" >> results/results.csv
else
    echo "bench_alloc not built (macOS only); on Linux count allocations with heaptrack, see the guide" >&2
fi

$PY scripts/make_table.py results/results.csv kalman-cpp --readme README.md
$PY scripts/make_charts.py results/results.csv docs/assets
