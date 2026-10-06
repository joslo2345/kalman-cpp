"""Usage: python scripts/gbench_to_csv.py results/gbench.json "commit,cpu,os,toolchain,date"

Appends the median time per step of each benchmark to results/results.csv.
"""
import csv
import json
import sys

data = json.load(open(sys.argv[1]))
env = sys.argv[2].split(",")
versions = {
    "kalman-cpp": env[0],
    "kalman-cpp-sqrt": env[0],
    "opencv": data["context"].get("opencv_version", "unknown"),
    "mherb-kalman": data["context"].get("mherb_kalman_commit", "unknown"),
}

with open("results/results.csv", "a", newline="") as f:
    out = csv.writer(f)
    for b in data["benchmarks"]:
        if b.get("aggregate_name") != "median":
            continue
        library, scenario, filt, precision = b["run_name"].split("/")[0].split("|")
        ns_per_step = 1e9 / b["items_per_second"]
        out.writerow([library, versions.get(library, "unknown"), scenario, filt, precision,
                      "time_per_step", f"{ns_per_step:.2f}", "ns", *env])
