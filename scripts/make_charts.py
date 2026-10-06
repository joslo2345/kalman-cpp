"""Usage: python scripts/make_charts.py results/results.csv docs/assets

Draws the README benchmark charts as static SVGs, one light and one dark
variant each (the README picks one with <picture> and prefers-color-scheme):
    bench-time-{light,dark}.svg       time per predict+update step (log scale)
    bench-stability-{light,dark}.svg  S4 steps survived out of 1,000,000

Colors follow the library, never its rank. The palette was checked with the
dataviz validator; light-mode aqua/yellow are below 3:1 contrast, so every mark
is also direct-labeled and the README keeps the full results table.
"""

import csv
import math
import sys
from pathlib import Path

LIBS = {  # display name, light color, dark color
    "kalman-cpp": ("kalman-cpp (Joseph KF)", "#2a78d6", "#3987e5"),
    "kalman-cpp-sqrt": ("kalman-cpp (square-root KF)", "#eb6834", "#d95926"),
    "mherb-kalman": ("mherb/kalman", "#1baf7a", "#199e70"),
    "opencv": ("OpenCV", "#eda100", "#c98500"),
    "naive": ("naive textbook KF", "#e87ba4", "#d55181"),
}
THEMES = {  # transparent background: these sit on GitHub's own surfaces
    "light": {"text": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0", "bg": "#ffffff", "col": 1},
    "dark": {"text": "#f0f6fc", "muted": "#9198a1", "grid": "#3d444d", "bg": "#0d1117", "col": 2},
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def load(path):
    out = {}
    for r in csv.DictReader(open(path, newline="")):
        out[(r["library"], r["scenario"], r["metric"])] = float(r["value"])
    return out


def svg(width, height, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="{FONT}" role="img" aria-label="{title}">\n'
            f"<title>{title}</title>\n{body}</svg>\n")


def text(x, y, s, fill, size=12, anchor="start", weight="normal", halo=None):
    # A halo in the page color keeps labels readable where they cross gridlines.
    h = f' stroke="{halo}" stroke-width="4" stroke-linejoin="round" paint-order="stroke"' if halo else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{h}>{s}</text>\n')


def fmt_ns(v):
    return f"{v / 1000:.2f} µs" if v >= 1000 else f"{v:.0f} ns"


def time_chart(data, theme):
    t = THEMES[theme]
    libs = ["kalman-cpp", "kalman-cpp-sqrt", "mherb-kalman", "opencv"]
    scenarios = [("S1", "S1 · 1-D tracking (2 states)"), ("S2", "S2 · 2-D tracking (4 states)"),
                 ("S5", "S5 · INS error state (15 states)")]
    W, left, right = 760, 210, 90
    x0, x1 = left, W - right
    lo, hi = 1.0, 4.0  # log10 ns: 10 ns .. 10 µs
    X = lambda v: x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)
    row_h, panel_gap, top = 26, 22, 92

    body = text(16, 28, "Time per predict + update step", t["text"], 16, weight="600")
    body += text(16, 48, "Lower is better · median of 30 runs · log scale · Apple M3 Pro, -O2", t["muted"], 12)
    lx = 16  # legend: always present for several series; rows are also labeled
    for lib in libs:
        name, cl, cd = LIBS[lib]
        body += f'<circle cx="{lx + 5}" cy="68" r="5" fill="{cl if t["col"] == 1 else cd}"/>\n'
        body += text(lx + 15, 72, name, t["text"], 12)
        lx += 15 + 7.2 * len(name) + 22

    y = top
    for scen, label in scenarios:
        body += text(16, y + 4, label, t["text"], 13, weight="600")
        y += 14
        y_axis_top = y
        y_axis_bottom = y + row_h * len(libs)
        for exp in range(int(lo), int(hi) + 1):  # recessive decade gridlines, under the marks
            gx = X(10**exp)
            body += (f'<line x1="{gx:.1f}" y1="{y_axis_top:.1f}" x2="{gx:.1f}" y2="{y_axis_bottom:.1f}" '
                     f'stroke="{t["grid"]}" stroke-width="1"/>\n')
        for lib in libs:
            v = data.get((lib, scen, "time_per_step"))
            name, cl, cd = LIBS[lib]
            color = cl if t["col"] == 1 else cd
            cy = y + row_h / 2
            body += text(left - 12, cy + 4, name, t["muted"], 12, anchor="end")
            body += (f'<line x1="{x0}" y1="{cy:.1f}" x2="{x1}" y2="{cy:.1f}" stroke="{t["grid"]}" '
                     f'stroke-width="1" stroke-dasharray="2 4"/>\n')
            if v is not None:
                body += f'<circle cx="{X(v):.1f}" cy="{cy:.1f}" r="6" fill="{color}"/>\n'
                body += text(X(v) + 11, cy + 4, fmt_ns(v), t["text"], 12, weight="600", halo=t["bg"])
            y += row_h
        y += panel_gap
    for exp, lab in zip(range(1, 5), ["10 ns", "100 ns", "1 µs", "10 µs"]):
        body += text(X(10**exp), y - panel_gap + 16, lab, t["muted"], 11, anchor="middle")
    H = int(y + 6)
    return svg(W, H, body, "Time per predict and update step, lower is better")


def stability_chart(data, theme):
    t = THEMES[theme]
    libs = ["kalman-cpp-sqrt", "kalman-cpp", "mherb-kalman", "naive", "opencv"]
    W, left, right = 760, 210, 40
    x0, x1 = left, W - right
    total = 1_000_000
    X = lambda v: x0 + v / total * (x1 - x0)
    row_h, bar_h, top = 34, 14, 78

    body = text(16, 28, "Steps survived out of 1,000,000 (S4, float32)", t["text"], 16, weight="600")
    body += text(16, 48, "Higher is better · vague prior (variance 1e6) vs precise measurements (variance 1e-6)",
                 t["muted"], 12)
    y = top
    for lib in libs:
        v = data.get((lib, "S4", "steps_to_failure"))
        if v is None:
            continue
        name, cl, cd = LIBS[lib]
        color = cl if t["col"] == 1 else cd
        cy = y + row_h / 2
        body += text(left - 12, cy + 4, name, t["muted"], 12, anchor="end")
        body += (f'<line x1="{x0}" y1="{cy:.1f}" x2="{x1}" y2="{cy:.1f}" stroke="{t["grid"]}" '
                 f'stroke-width="1" stroke-dasharray="2 4"/>\n')
        if v >= total:
            w = X(v) - x0
            # 4px rounded data end, square at the baseline.
            body += (f'<path d="M{x0},{cy - bar_h / 2:.1f} h{w - 4:.1f} q4,0 4,4 v{bar_h - 8:.1f} '
                     f'q0,4 -4,4 h{-(w - 4):.1f} z" fill="{color}"/>\n')
            body += text(x0 + 10, cy + 4.5, "never failed", "#ffffff", 12, weight="600")
        else:
            body += f'<circle cx="{x0 + 4}" cy="{cy:.1f}" r="5" fill="{color}"/>\n'
            body += text(x0 + 16, cy + 4.5, f"lost a valid covariance at step {int(v)}", t["text"], 12)
        y += row_h
    body += (f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{y}" stroke="{t["muted"]}" stroke-width="1"/>\n')
    for frac, lab in [(0, "0"), (0.5, "500k"), (1, "1M")]:
        body += text(X(frac * total), y + 16, lab, t["muted"], 11, anchor="middle")
    return svg(W, int(y + 26), body, "Steps survived out of one million in float32, higher is better")


def main():
    data = load(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (out / f"bench-time-{theme}.svg").write_text(time_chart(data, theme))
        (out / f"bench-stability-{theme}.svg").write_text(stability_chart(data, theme))
    print(f"wrote 4 charts to {out}")


if __name__ == "__main__":
    main()
