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
WIDTH, HEIGHT = 1600, 1600
X0, X1 = 125, 1505
Y0, Y1 = 255, 670
SCORE_MIN, SCORE_MAX = .850, .990
START = datetime.fromisoformat("2026-09-27T08:00:00-07:00")
END = datetime.fromisoformat("2026-11-16T18:00:00+00:00")


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
    '<desc id="desc">Early single-model Probe scores, blend CV through November, and a separate recovered qualifying audit diamond on a continuous scale.</desc>',
    '<rect width="1600" height="800" fill="#10131d"/>',
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


line(X0, y_at(.9525), X1, y_at(.9525), "#70d9d0", 2)
text(X0, y_at(.9525)-18, "0.9525", "#70d9d0", 22, weight="700")
text(X1, y_at(.9525)-18, "Cinematch baseline · Netflix Test", "#70d9d0", 22, "end", "700")
text(X0, 221, "PROBE MODELS, BLEND CV + QUALIFYING AUDIT", "#a8b0c2", 21, weight="700")
icon("codex", 1242, 207, 24)
text(1265, 215, "Codex", "#f2f1ed", 22)
icon("claude", 1393, 207, 24)
text(1415, 215, "Claude", "#f2f1ed", 22)

for tick in (.850, .875, .900, .925, .975):
    y = y_at(tick)
    line(X0, y, X1, y, "#4b5364", 1, .55)
    text(X0 - 16, y + 7, f"{tick:.3f}", "#a8b0c2", 20, "end")

dates = [
    ("2026-09-27", "Sep 27"), ("2026-10-07", "Oct 7"),
    ("2026-10-17", "Oct 17"), ("2026-10-27", "Oct 27"),
    ("2026-11-06", "Nov 6"), ("2026-11-15", "Nov 15"),
]
for date, label in dates:
    x = x_at(date + "T12:00:00-07:00")
    x = min(x, X1 - 8)
    line(x, Y0, x, Y1, "#4b5364", 1, .2)
    text(x, 710, label, "#a8b0c2", 21, "middle")

for ident, score, date, _strategy, metric, credit, _status, _title, _note, _source, _basis in MILESTONES:
    if _status == "screen":
        svg.append('<g opacity=".4">')
    if metric == "qualifying":
        x, y = x_at(date), y_at(float(score))
        svg.append(f'<path d="M{x} {y-10} L{x+10} {y} L{x} {y+10} L{x-10} {y} Z" fill="#baaaff"/>')
    else:
        icon(credit, x_at(date), y_at(float(score)), 23 if credit == "claude" else 21)
    if _status == "screen":
        svg.append("</g>")

for ident, label, dx, dy, anchor in (
    ("B1", "B1 · 0.89369", 18, -12, "start"),
    ("PUB466 NN2", "Probe CV · 0.8645338", -18, -15, "end"),
    ("QUAL619 ridge", "Recovered audit · 0.8571903", -18, 28, "end"),
):
    row = next(m for m in MILESTONES if m[0] == ident)
    text(x_at(row[2]) + dx, y_at(float(row[1])) + dy, label, "#f2f1ed", 20, anchor, "700")

line(X0, y_at(.85725), X1, y_at(.85725), "#baaaff", 2)
text(X0, y_at(.85725)-18, "0.85725", "#baaaff", 22, weight="700")
text(X1, y_at(.85725)-18, "Netflix 10% improvement goal · Test RMSE", "#baaaff", 22, "end", "700")
text(X0, 760, "Probe CV and the recovered qualifying audit use different rows and training. Diamond: adaptive community audit, not an official Test score.", "#a8b0c2", 19)
audit_svg = runpy.run_path(str(ROOT / "scripts/build_research_journey.py"))["qualifying_svg"]()
svg.append('<rect x="0" y="800" width="1600" height="800" fill="#10131d"/>')
text(94, 857, "NOVEMBER BREAKTHROUGH · RECOVERED QUALIFYING AUDIT", "#70d9d0", 28, weight="700")
text(94, 902, "619-input ridge: 0.8571903481 · 10.00626267% below 0.9525", "#f2f1ed", 27, weight="700")
svg.append(audit_svg.replace('<svg ', '<svg x="80" y="930" width="1440" height="660" ', 1))
svg.append("</svg>")
OUT_SVG.write_text("\n".join(svg) + "\n")

if shutil.which("sips"):
    subprocess.run(["sips", "-s", "format", "png", str(OUT_SVG), "--out", str(OUT_PNG)], check=True, capture_output=True)
    print(f"Wrote {OUT_SVG.relative_to(ROOT)} and {OUT_PNG.relative_to(ROOT)}")
else:
    print(f"Wrote {OUT_SVG.relative_to(ROOT)}; convert it to PNG for the README")
