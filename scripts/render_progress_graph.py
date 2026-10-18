#!/usr/bin/env python3
"""Render the README score preview from the research journey milestones.

The SVG is portable; on macOS, sips also writes the PNG embedded in README.
"""

from datetime import datetime
from html import escape
from pathlib import Path
import runpy
import shutil
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
MILESTONES = runpy.run_path(str(ROOT / "scripts/build_research_journey.py"))["MILESTONES"]
OUT_SVG = ROOT / "assets/progress_graph.svg"
OUT_PNG = ROOT / "assets/progress_graph.png"
WIDTH, HEIGHT = 1600, 900
X0, X1 = 125, 1505
Y0, Y1 = 255, 670
SCORE_MIN, SCORE_MAX = .865, .895
START = datetime.fromisoformat("2026-09-27T08:00:00-07:00")
END = datetime.fromisoformat("2026-10-11T09:30:00-07:00")


def x_at(value):
    elapsed = (datetime.fromisoformat(value) - START).total_seconds()
    full = (END - START).total_seconds()
    return X0 + elapsed / full * (X1 - X0)


def y_at(score):
    return Y1 - (score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN) * (Y1 - Y0)


def icon_path(name):
    root = ET.parse(ROOT / "assets" / ("claude-code.svg" if name == "claude" else "codex.svg")).getroot()
    path = next(node for node in root.iter() if node.tag.endswith("path"))
    return path.attrib["d"]


ICONS = {name: icon_path(name) for name in ("claude", "codex")}
COLOR = {"claude": "#D97757", "codex": "#BAAAFF"}
svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
    '<title id="title">Can AI win the $1M Netflix Challenge</title>',
    '<desc id="desc">Five-user-fold Probe CV scores from B1 through R10, with Codex and Claude logos, and separate Cinematch and ten-percent Test reference bands.</desc>',
    '<rect width="1600" height="900" fill="#10131d"/>',
    '<rect x="92" y="232" width="1435" height="460" rx="16" fill="#181d29"/>',
    '<text x="94" y="70" fill="#f2f1ed" font-family="Arial,Helvetica,sans-serif" font-size="43" font-weight="700">Can AI win the $1M Netflix Challenge</text>',
    '<text x="96" y="110" fill="#a8b0c2" font-family="Arial,Helvetica,sans-serif" font-size="23">Score progress by calendar date · lower RMSE is better</text>',
]


def text(x, y, value, color="#a8b0c2", size=22, anchor="start", weight="400"):
    svg.append(
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-family="Arial,Helvetica,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def line(x1, y1, x2, y2, color, width=1, opacity=1):
    svg.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}" opacity="{opacity}"/>'
    )


def icon(name, x, y, size=23):
    scale = size / 24
    svg.append(
        f'<g transform="translate({x-size/2:.2f} {y-size/2:.2f}) scale({scale:.5f})">'
        f'<path d="{escape(ICONS[name])}" fill="{COLOR[name]}" fill-rule="evenodd"/></g>'
    )


line(X0, 169, X1, 169, "#70d9d0", 2)
text(X0, 150, "0.9525", "#70d9d0", 22, weight="700")
text(X1, 150, "Cinematch baseline · Netflix Test", "#70d9d0", 22, "end", "700")
text(X0, 221, "FIVE-USER-FOLD PROBE CV", "#a8b0c2", 21, weight="700")
icon("codex", 1242, 207, 24)
text(1265, 215, "Codex", "#f2f1ed", 22)
icon("claude", 1393, 207, 24)
text(1415, 215, "Claude", "#f2f1ed", 22)

for tick in (.865, .870, .875, .880, .885, .890, .895):
    y = y_at(tick)
    line(X0, y, X1, y, "#4b5364", 1, .55)
    text(X0 - 16, y + 7, f"{tick:.3f}", "#a8b0c2", 20, "end")

dates = [
    ("2026-09-27", "Sep 27"),
    ("2026-09-30", "Sep 30"),
    ("2026-10-03", "Oct 3"),
    ("2026-10-06", "Oct 6"),
    ("2026-10-09", "Oct 9"),
    ("2026-10-11", "Oct 11"),
]
for date, label in dates:
    x = x_at(date + "T12:00:00-07:00")
    x = min(x, X1 - 8)
    line(x, Y0, x, Y1, "#4b5364", 1, .2)
    text(x, 814, label, "#a8b0c2", 21, "middle")

for ident, score, date, _strategy, metric, credit, _status, _title, _note, _source, _basis in MILESTONES:
    if metric == "single":
        continue
    icon(credit, x_at(date), y_at(float(score)), 23 if credit == "claude" else 21)

for ident, label, dx, dy, anchor in (
    ("B1", "B1 · 0.89369", 18, -12, "start"),
    ("B6 tree", "B6 · 0.88040", 20, -8, "start"),
    ("C12 parallel", "C12 · 0.87478", 20, 2, "start"),
    ("R7-MT56-s3", "R7 · 0.87039", 20, 2, "start"),
    ("R9 LAT", "R9 · 0.86687", -18, -19, "end"),
):
    row = next(m for m in MILESTONES if m[0] == ident)
    text(x_at(row[2]) + dx, y_at(float(row[1])) + dy, label, "#f2f1ed", 20, anchor, "700")

line(X0, 754, X1, 754, "#baaaff", 2)
text(X0, 738, "0.8572", "#baaaff", 22, weight="700")
text(X1, 738, "Netflix 10% improvement goal · Test RMSE", "#baaaff", 22, "end", "700")
text(X0, 860, "Test reference lines and Probe CV points use different evaluation sets. C5/C12 are parallel-branch results; R10 is an unpromoted screen.", "#a8b0c2", 19)
svg.append("</svg>")
OUT_SVG.write_text("\n".join(svg) + "\n")

if shutil.which("sips"):
    subprocess.run(["sips", "-s", "format", "png", str(OUT_SVG), "--out", str(OUT_PNG)], check=True, capture_output=True)
    print(f"Wrote {OUT_SVG.relative_to(ROOT)} and {OUT_PNG.relative_to(ROOT)}")
else:
    print(f"Wrote {OUT_SVG.relative_to(ROOT)}; convert it to PNG for the README")
