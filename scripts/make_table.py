"""Usage: python scripts/make_table.py results/results.csv <our-library-name> [--readme README.md]

Prints the comparison table. With --readme, also replaces the table between
the <!-- BENCH:START --> and <!-- BENCH:END --> markers in that file.
"""
import csv
import re
import sys
from collections import defaultdict

LOWER_IS_BETTER = {"time_per_step", "cycles_per_step", "rmse", "heap_allocations",
                   "peak_memory", "flash_bytes", "ram_bytes"}
HIGHER_IS_BETTER = {"steps_to_failure"}

args = sys.argv[1:]
readme = None
if "--readme" in args:
    i = args.index("--readme")
    readme = args[i + 1]
    del args[i:i + 2]

rows = list(csv.DictReader(open(args[0], newline="")))
ours = args[1]
mine = sorted({r["library"] for r in rows if r["library"].startswith(ours)})
libs = mine + sorted({r["library"] for r in rows} - set(mine))

cells = defaultdict(dict)
for r in rows:
    key = (r["scenario"], r["filter"], r["precision"], r["metric"], r["unit"])
    cells[key][r["library"]] = float(r["value"])


def ratio(metric, vals):
    # Libraries named "<ours>-something" are our own variants (e.g. the
    # square-root filter); "ours" is the best of them for each row.
    mine = [v for lib, v in vals.items() if lib.startswith(ours)]
    others = [v for lib, v in vals.items() if not lib.startswith(ours)]
    if not mine or not others:
        return "n/a"
    if metric in LOWER_IS_BETTER:
        best, me = min(others), min(mine)
        if me == 0:
            return "–" if best == 0 else "∞"
        return f"{best / me:.2f}x"
    if metric in HIGHER_IS_BETTER:
        best, me = max(others), max(mine)
        if best == 0:
            return "∞" if me > 0 else "–"
        return f"{me / best:.2f}x"
    return "–"  # nees, max_abs_diff: read the values directly


lines = ["| Scenario | Filter | Precision | Metric | " + " | ".join(libs) + " | Ours vs best other |",
         "|" + "---|" * (len(libs) + 5)]
for (scen, filt, prec, metric, unit), vals in sorted(cells.items()):
    values = [f"{vals[lib]:.4g}" if lib in vals else "n/a" for lib in libs]
    lines.append(f"| {scen} | {filt} | {prec} | {metric} ({unit}) | " + " | ".join(values)
                 + f" | {ratio(metric, vals)} |")
table = "\n".join(lines)
print(table)

if readme:
    text = open(readme).read()
    new, n = re.subn(r"(<!-- BENCH:START -->\n).*?(\n<!-- BENCH:END -->)",
                     lambda m: m.group(1) + table + m.group(2), text, flags=re.S)
    if n != 1:
        sys.exit(f"{readme}: expected exactly one BENCH:START/BENCH:END block")
    open(readme, "w").write(new)
