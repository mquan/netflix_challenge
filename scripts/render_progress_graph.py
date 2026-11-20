#!/usr/bin/env python3
"""Render the README score preview from the research journey milestones.

The SVG is portable; on macOS, sips also writes the PNG embedded in README.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from html import escape
from pathlib import Path
import runpy
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
JOURNEY = runpy.run_path(str(ROOT / "scripts/build_research_journey.py"))
JOURNEY["validate_fresh_status"]()  # Refuse to render unverified current score claims.
MILESTONES = JOURNEY["own_training_milestones"](JOURNEY["MILESTONES"])
STATUS = json.loads((ROOT / "experiments/fresh-reproduction-status.json").read_text())
CURRENT_CV = next(row for row in MILESTONES if row[4] == "fresh_cv" and float(row[1]) == STATUS["best_development_rmse"])
CURRENT_QUAL = next(row for row in MILESTONES if row[9] == STATUS["latest_qualifying_verification_path"])
SELECTION = JOURNEY["overview_selection"](MILESTONES, STATUS)
PLOT_MILESTONES = [row for row in MILESTONES
    if datetime.fromisoformat(row[2]).astimezone(ZoneInfo('America/Los_Angeles')).date().isoformat() not in SELECTION['days']
    or row[0] in SELECTION['ids']]
QUAL_GROUPS = STATUS["latest_qualifying_groups"]
QUAL_WORST = max(row["rmse"] for row in QUAL_GROUPS.values())
if abs(float(CURRENT_QUAL[1])-QUAL_WORST)>1e-12:
    raise ValueError("Latest qualifying plot point differs from the verified record")
OUT_SVG = ROOT / "assets/progress_graph.svg"
OUT_PNG = ROOT / "assets/progress_graph.png"
WIDTH, HEIGHT = 1600, 1600
X0, X1 = 125, 1505
Y0, Y1 = 255, 670
SCORE_MIN, SCORE_MAX = .850, .990
START = datetime.fromisoformat("2026-09-27T08:00:00-07:00")
END = max(datetime.fromisoformat("2026-11-19T20:00:00+00:00"), max(datetime.fromisoformat(row[2]) for row in MILESTONES) + timedelta(minutes=30))


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
    '<desc id="desc">Early single-model Probe scores, blend CV through November, and own-trained qualifying results on a continuous scale.</desc>',
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
    ("2026-11-06", "Nov 6"), ("2026-11-19", "Nov 19"),
]
for date, label in dates:
    x = x_at(date + "T12:00:00-07:00")
    x = min(x, X1 - 8)
    line(x, Y0, x, Y1, "#4b5364", 1, .2)
    text(x, 710, label, "#a8b0c2", 21, "middle")

for ident, score, date, _strategy, metric, credit, _status, _title, _note, _source, _basis in PLOT_MILESTONES:
    if _status == "screen":
        svg.append('<g opacity=".4">')
    if metric == "fresh_qualifying":
        x, y = x_at(date), y_at(float(score))
        icon("codex", x, y, 21)
    elif metric == "qualifying":
        x, y = x_at(date), y_at(float(score))
        svg.append(f'<path d="M{x} {y-10} L{x+10} {y} L{x} {y+10} L{x-10} {y} Z" fill="#baaaff"/>')
    else:
        icon(credit, x_at(date), y_at(float(score)), 23 if credit == "claude" else 21)
    if _status == "screen":
        svg.append("</g>")

for ident, label, dx, dy, anchor in (
    ("B1", "B1 · 0.89369", 18, -12, "start"),
    ("H8 full", "Original-Probe CV · 0.8666488", -18, -15, "end"),
    (CURRENT_CV[0], f"Own training CV · {STATUS['best_development_rmse']:.7f}", -18, -68, "end"),
    (CURRENT_QUAL[0], f"Own qualifying audit · {QUAL_WORST:.7f}", -18, -130, "end"),
):
    row = next(m for m in MILESTONES if m[0] == ident)
    text(x_at(row[2]) + dx, y_at(float(row[1])) + dy, label, "#f2f1ed", 20, anchor, "700")

line(X0, y_at(.85725), X1, y_at(.85725), "#baaaff", 2)
text(X0, y_at(.85725)+34, "0.85725", "#baaaff", 22, weight="700")
text(X1, y_at(.85725)+34, "Historical Netflix Test target · 0.85725", "#baaaff", 22, "end", "700")
text(X0, 760, "Own qualifying audits use the Codex logo. Recovered labels; different rows from development CV.", "#a8b0c2", 19)
svg.append('<rect x="0" y="800" width="1600" height="800" fill="#10131d"/>')
text(94, 857, "REVISED PROMOTIONS · ORIGINAL-PROBE DEVELOPMENT", "#70d9d0", 28, weight="700")
text(94, 905, "Lower overall RMSE promotes a candidate; the record line follows successive best scores.", size=23)
text(94, 952, "Filled cyan: new record     Hollow gold: promoted against its control, not a new record", size=22)
selected=[next(m for m in MILESTONES if m[0]==name) for name in ['R10 B5','HR2a','HR2b','LC6','N1 full','H8 full']]
zoom_y=lambda value:1370-(value-.8664)/.0005*340
for tick in [.8664,.8665,.8666,.8667,.8668,.8669]:
    line(X0,zoom_y(tick),X1,zoom_y(tick),'#4b5364',1,.5)
    text(X0-16,zoom_y(tick)+6,f'{tick:.4f}',size=19,anchor='end')
record_points=[]
for i,row in enumerate(selected):
    xx=X0+65+i*(X1-X0-130)/5;yy=zoom_y(float(row[1]))
    if row[6]=='promoted':record_points.append((xx,yy))
if record_points:
    svg.append('<polyline points="'+' '.join(f'{xx:.1f},{yy:.1f}' for xx,yy in record_points)+'" fill="none" stroke="#70d9d0" stroke-width="3"/>')
for i,row in enumerate(selected):
    xx=X0+65+i*(X1-X0-130)/5;yy=zoom_y(float(row[1]));record=row[6]=='promoted'
    svg.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="8" fill="{"#70d9d0" if record else "#10131d"}" stroke="{"#70d9d0" if record else "#f6c26b"}" stroke-width="3"/>')
    text(xx,yy-20,f'{float(row[1]):.7f}',color='#f2f1ed',size=20,anchor='middle')
    text(xx,1410,row[0],color='#f2f1ed',size=21,anchor='middle',weight='700')
text(94,1460,'Recorded experiment order · shared original-Probe rows · historical reports and original decisions preserved',size=21)
text(94,1510,f"Current own-training CV best: {STATUS['best_development_rmse']:.7f} on different known-ProbeX rows.",color='#f2f1ed',size=25,weight='700')
text(94,1555,f"Best own qualifying audit: {QUAL_GROUPS['source0']['rmse']:.7f} / {QUAL_GROUPS['source1']['rmse']:.7f} · {100*(1-QUAL_WORST/.9525):.4f}% on the worse group; 10% target missed.",size=22)
svg.append("</svg>")
OUT_SVG.write_text("\n".join(svg) + "\n")

if shutil.which("sips"):
    subprocess.run(["sips", "-s", "format", "png", str(OUT_SVG), "--out", str(OUT_PNG)], check=True, capture_output=True)
    print(f"Wrote {OUT_SVG.relative_to(ROOT)} and {OUT_PNG.relative_to(ROOT)}")
else:
    print(f"Wrote {OUT_SVG.relative_to(ROOT)}; convert it to PNG for the README")
