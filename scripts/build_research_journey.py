#!/usr/bin/env python3
"""Build the self-contained research journey from documented milestones and run events."""
import base64
import datetime as dt
import json
from html import escape
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"

# These are editorial milestones, not a claim that every experiment was a new best.
# Calendar dates before timestamped run events are documentation upper bounds.
MILESTONES = [
    ("C0", ".98256528", "2026-09-27T08:45:41-07:00", "Biases", "single", "codex", "promoted", "User and movie bias", "The first project model proved ingestion and probe isolation, but it missed most user–movie interaction.", "RESULTS.md", "commit upper bound"),
    ("C1", ".93491601", "2026-09-27T08:45:41-07:00", "Time", "single", "codex", "promoted", "Temporal effects model", "Day effects, frequency and drift delivered the largest early jump.", "RESULTS.md", "commit upper bound"),
    ("C2", ".90267216", "2026-09-27T09:10:33-07:00", "Factors", "single", "codex", "promoted", "32-factor SVD++", "Collaborative factors captured interactions that biases could not.", "RESULTS.md", "commit upper bound"),
    ("B1", ".8936900925", "2026-09-27T09:23:40-07:00", "Blend", "cv", "codex", "promoted", "First user-fold blend", "User-disjoint fitting gave a cleaner development estimate for combining diverse factor columns.", "RESULTS.md", "log time proxy"),
    ("B4", ".8866646513", "2026-09-27T16:08:00-07:00", "Neighborhood", "cv", "codex", "promoted", "Learned item neighbors", "A weaker standalone neighbor model added complementary local movie signal.", "RESULTS.md", "log time proxy"),
    ("B7", ".8858460517", "2026-09-27T16:22:00-07:00", "Neighborhood", "cv", "codex", "promoted", "128-neighbor graph", "More neighbor capacity lowered the five-fold blend score again.", "RESULTS.md", "log time proxy"),
    ("B6 tree", ".8804020935", "2026-09-27T18:08:50-07:00", "Blend", "cv", "claude", "promoted", "LightGBM over 15 columns", "Residual kNN, RBM and AutoRec columns plus support metadata made the nonlinear blender useful.", "RESULTS.md", "observed run finish"),
    ("Union", ".8782199562", "2026-09-27T18:30:47-07:00", "Blend", "cv", "codex", "promoted", "Two branches, fixed average", "A 50/50 average of complementary banks and blenders gained without fitting a second-level weight.", "COMBINED_BRANCHES.md", "commit upper bound"),
    ("R3", ".8773860815", "2026-09-27T19:28:07-07:00", "Blend", "cv", "codex", "promoted", "Residual tree correction", "Predicting residual error directly beat the previous equal blend.", "ROUND3.md", "commit upper bound"),
    ("R4-C", ".8754534170", "2026-09-27T20:20:03-07:00", "Neighborhood", "cv", "claude", "promoted", "Wider neighborhood bank", "Time-decayed and 256-neighbor kNN columns plus converged models improved every fold.", "ROUND4.md", "commit upper bound"),
    ("R5-meta", ".8748094089", "2026-09-27T21:56:15-07:00", "Metadata", "cv", "claude", "promoted", "Training-derived context", "Seven user and movie context features sharpened the residual tree.", "ROUND5.md", "commit upper bound"),
    ("R6-RBM5-s3", ".8731159148", "2026-09-28T13:45:25-07:00", "RBM", "cv", "claude", "promoted", "Six conditional RBM columns", "Different hidden sizes and seeds paid more than further continuation of one RBM.", "ROUND6.md", "commit upper bound"),
    ("C5 parallel", ".8759027924", "2026-09-30T21:52:13-07:00", "GP", "cv_parallel", "codex", "parallel", "Local GP compact blend", "The parallel Codex branch rebuilt a compact 17-column bank and added a Gaussian-process prediction. C5 was its best reproducible local result at that checkpoint, but it did not supersede the main score frontier.", "GP_COMPACT.md", "run-start upper bound"),
    ("R7-MH3", ".8725848", "2026-09-30T22:31:12-07:00", "Sequence", "cv", "claude", "promoted", "Masked rating history", "A categorical masked-history predictor brought sequence signal into the bank.", "ROUND7.md", "commit upper bound"),
    ("R7-MT56-s3", ".8703893", "2026-10-01T02:07:55-07:00", "Sequence", "cv", "claude", "promoted", "Masked history variants", "Multiple temporal masked models and three tree seeds pushed below 0.871.", "ROUND7.md", "commit upper bound"),
    ("C12 parallel", ".8747780516", "2026-10-01T21:12:19-07:00", "Session context", "cv_parallel", "codex", "parallel", "Fixed session-context average", "The parallel Codex branch combined C10 and C11 after GP, kNN, RBM and training-day context experiments. C12 narrowly beat historical R5 overall, but only three of five folds improved and the main Round 7 bank was already lower.", "SESSION_META.md", "commit upper bound"),
    ("R8 Bayes", ".8692319", "2026-10-03T14:49:11-07:00", "Factors", "cv", "claude", "promoted", "Bayesian timeSVD++", "A rank-128 Bayesian factor snapshot added a distinct posterior-mean column.", "ROUND8.md", "commit upper bound"),
    ("R8 NN2", ".8676503", "2026-10-04T19:50:33-07:00", "Blend", "cv", "claude", "promoted", "Neural residual blend", "A fixed 50/50 LightGBM and NN2 average improved all five user folds.", "ROUND8.md", "commit upper bound"),
    ("R8 A2e", ".8674131", "2026-10-06T19:26:12-07:00", "Sequence", "cv", "claude", "promoted", "Longer masked sequence", "The 768-step history model and drift factors advanced the promoted result.", "ROUND8.md", "commit upper bound"),
    ("R8 A2h", ".8670984524", "2026-10-09T04:26:14-07:00", "Sequence", "cv", "codex", "promoted", "1536-step A2h + NN2", "A2h improved every fold over the previous promoted ensemble.", "ROUND8.md", "observed run finish"),
    ("R9 LAT", ".8668706425", "2026-10-10T10:46:00-07:00", "Metadata", "cv", "codex", "promoted", "K1 latent and bias features", "Twenty-two features enriched the existing 79-column bank; every fold improved.", "ROUND9.md", "documented completion"),
    ("R10 B5", ".8668212", "2026-10-11T01:42:00-07:00", "Factors", "cv", "claude", "screen", "Bayesian item-frequency screen", "The mean improved by roughly 0.00005, but folds 0 and 1 worsened, so B5 was not promoted.", "ROUND10.md", "documented completion"),
]

RUNS = [
    ("T2-f64-lr005-reg015-d97", "Joint factors", "probe_rmse", "epoch", "Time-aware factor training", "claude"),
    ("R2-rbm100-fast", "RBM", "probe_rmse", "epoch", "Conditional RBM learning", "claude"),
    ("P2-rbm200-condbias-scratch", "RBM + context", "probe_rmse", "epoch", "Contextual RBM continuation", "claude"),
    ("B3-full-r128", "Bayesian factors", "mean_probe_rmse", "sweep", "Posterior-mean factor learning", "claude"),
]

# Selected public Quiz leaderboard observations, not a continuous reconstruction.
# Dates for annual prizes are the dated announcement/report, not necessarily submission time.
COMPETITION = [
    dict(date="2006-10-02", score=.9514, team="Cinematch", title="The public baseline",
         note="Netflix launched the challenge with Cinematch at 0.9514 RMSE on the public Quiz set.",
         source="https://gregstoll.com/~gregstoll/netflixprize/leaderboard20061019.html"),
    dict(date="2006-10-18", score=.9084, team="How low can he go?", title="First weeks: 0.9084",
         note="A preserved October 2006 Netflix leaderboard lists this leading submission on October 18.",
         source="https://gregstoll.com/~gregstoll/netflixprize/leaderboard20061019.html"),
    dict(date="2007-11-13", score=.8712, team="KorBell / BellKor", title="2007 Progress Prize",
         note="BellKor blended 107 results; the 0.8712 Quiz score earned the first $50,000 progress prize. Date is the prize announcement.",
         source="https://pzs.dstu.dp.ua/DataMining/recom/bibl/ProgressPrize2007_KorBell.pdf"),
    dict(date="2008-11-25", score=.8616, team="BellKor in BigChaos", title="2008 Progress Prize",
         note="The combined BellKor and BigChaos solution reached 0.8616. Date is the team's report, not its submission.",
         source="https://www2.seas.gwu.edu/~simhaweb/champalg/cf/papers/ProgressPrize2008_BigChaos.pdf"),
    dict(date="2009-06-26", score=.8558, team="BellKor's Pragmatic Chaos", title="The 10% line falls",
         note="The new combined team crossed the public Quiz threshold and started the final 30-day period.",
         source="https://www.stat.berkeley.edu/~jon/stat-151b-spring-2012/netflix_overview.pdf"),
    dict(date="2009-07-26", score=.8553, team="The Ensemble", title="Final public leaderboard",
         note="The Ensemble led the public Quiz board at 0.8553. BellKor's Pragmatic Chaos had 0.8554 on Quiz; both rounded to 0.8567 on hidden Test, and the earlier submission won.",
         source="https://people.csail.mit.edu/dsontag/courses/ml12/slides/lecture12.pdf"),
]

# Score or run-recording commits with an explicit Claude co-author trailer.
CLAUDE_RECORDS = {
    "B6 tree": "325366559b2e4d6395e160d8c47ac8386b805730",
    "R4-C": "909ff0305d0093b115acdd0c9be9ba501c224eab",
    "R5-meta": "1314dfd4304ee1626b200363e543999f06ef7a3f",
    "R6-RBM5-s3": "5113022c841f8f877f2a28762ee9809b26ef5065",
    "R7-MH3": "3b6d035796afa003dc36defc5ebd131d13ac0a15",
    "R7-MT56-s3": "f2d13c52ffc2c59308d28f8b5960407b0a791e64",
    "R8 Bayes": "14b2540dd4488aefe1721804ab9aa42126833a88",
    "R8 NN2": "e90adad1d222afe0b7d120c683c5083c81cff2c5",
    "R8 A2e": "16f4eb4a385aa5a0347abf54af2eb5edd2b80fda",
    "R10 B5": "d9796da1b826e5ed5e919e01a18823d63d300d0a",
    "T2-f64-lr005-reg015-d97": "bcdf4a27b389337d243ed53a2c7ee20e3a9f0d6b",
    "R2-rbm100-fast": "325366559b2e4d6395e160d8c47ac8386b805730",
    "P2-rbm200-condbias-scratch": "2424aaee6773927b1992ecd8590f9fed6d8196a8",
    "B3-full-r128": "2d4488ed023f15b23784292784b96a00611e0aac",
}

# Quan-only score commits; the user confirmed these were Codex sessions.
CODEX_RECORDS = {
    "C0": "4e7f65636a8e3f448e591cb8490ebcb56d513d34",
    "C1": "4e7f65636a8e3f448e591cb8490ebcb56d513d34",
    "C2": "66591d002dbaa4a45524457e162ce49ffa255c29",
    "B1": "cffd6bfe8406ba3332511e7ba26da0c0ca23b874",
    "B4": "cffd6bfe8406ba3332511e7ba26da0c0ca23b874",
    "B7": "700fbeb9af146e73c4d4c0f7387c0e0d5a9581e2",
    "Union": "bfb875f26de3de7895ab118615e8156651f1e28a",
    "R3": "43da9c70ffae34a0c33c8643799f671ea396a271",
    "R8 A2h": "f1639454e15b0637f61537b993f7f9de8151b3d7",
    "R9 LAT": "0e8171bff83bb63d9beae1d9fece06ffba199dae",
    "C5 parallel": "c368d4bfca0f56ed02dfad5a41dc97e10fa9be85",
    "C12 parallel": "c368d4bfca0f56ed02dfad5a41dc97e10fa9be85",
}

RECORD_COMMITS = {**CLAUDE_RECORDS, **CODEX_RECORDS}


def run_curve(name, label, metric, step_key, description, credit):
    manifest = json.loads((EXPERIMENTS / (name + ".json")).read_text())
    start = dt.datetime.fromisoformat(manifest["started_utc"])
    points = []
    for raw in (EXPERIMENTS / (name + ".events.jsonl")).read_text().splitlines():
        event = json.loads(raw)
        line = event["line"]
        score = re.search(r"(?<!\w)" + metric + r"=([0-9.]+)", line)
        step = re.search(r"(?<!\w)" + step_key + r"=(\d+)", line)
        if score and step:
            hour = (dt.datetime.fromisoformat(event["observed_utc"]) - start).total_seconds() / 3600
            points.append([round(hour, 4), round(float(score.group(1)), 7), int(step.group(1))])
    if not points:
        raise ValueError(f"No runtime score events for {name}")
    return dict(id=name, label=label, metric=metric, step=step_key, description=description,
                credit=credit, creditCommit=RECORD_COMMITS.get(name),
                elapsedHours=round(manifest["elapsed_seconds"] / 3600, 3), points=points,
                source=f"experiments/{name}.events.jsonl")


# Same recovered-label population throughout; never mix these with Probe CV.
QUALIFYING = [
    ("388-input ridge", "01:03", .8582647011016761, .8574231002840432),
    ("58-input ridge", "02:19", .8578789678524765, .8567927057358324),
    ("58-input NN2", "02:22", .8578011759258087, .8567189009050679),
    ("83-input ridge", "02:31", .8575942363558479, .8565118571755685),
    ("83-input NN2", "02:33", .8575337426262905, .8564604689103762),
    ("619-input ridge", "02:42", .8571903480630092, .8561542082012507),
]


def qualifying_svg():
    """One audited score sequence, reused by the site and README preview."""
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 440" role="img" aria-labelledby="audit-title audit-desc">',
             '<title id="audit-title">Recovered qualifying audit: both groups cross 0.85725</title>',
             '<desc id="audit-desc">Six candidate audits on November 16 UTC. Source1 falls from 0.8582647 to 0.8571903; source0 from 0.8574231 to 0.8561542. Adaptive community evaluation, not official Netflix scoring. Horizontal positions are candidate order.</desc>',
             '<rect width="960" height="440" fill="#181d29"/>']
    def text(x, y, value, color="#a8b0c2", size=14, anchor="start"):
        parts.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" text-anchor="{anchor}">{escape(value)}</text>')
    def y(score):
        return 325-(score-.8560)/.0025*255
    text(80, 30, "Recovered qualifying RMSE · lower is better", "#f2f1ed", 18)
    text(80, 53, "Candidate order · audit times: Nov 16 UTC / Nov 15 Pacific", size=13)
    for tick in (.8560, .8565, .8570, .8575, .8580, .8585):
        yy=y(tick)
        parts.append(f'<path d="M80 {yy} H875" stroke="#4b5364" opacity=".5"/>')
        text(70, yy+5, f"{tick:.4f}", size=13, anchor="end")
    parts.append(f'<path d="M80 {y(.85725)} H875" stroke="#baaaff" stroke-dasharray="6 5"/>')
    text(875, y(.85725)-9, "10% target: 0.85725", "#baaaff", 14, "end")
    for group, color, label in ((2, "#70d9d0", "source1"), (3, "#f6c26b", "source0")):
        coords=[(100+i*150, y(row[group])) for i,row in enumerate(QUALIFYING)]
        points=" ".join(f"{x},{yy}" for x,yy in coords)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for (x,yy),row in zip(coords,QUALIFYING):
            parts.append(f'<circle cx="{x}" cy="{yy}" r="5" fill="{color}"><title>{escape(row[0])} · {label} · {row[group]:.10f} · {row[1]} UTC</title></circle>')
        text(850, coords[-1][1]+24, f"{label}: {QUALIFYING[-1][group]:.7f}", color, 15, "end")
    for i,(name,time,*_) in enumerate(QUALIFYING):
        text(100+i*150, 358, name, "#f2f1ed", 13, "middle")
        text(100+i*150, 378, time+" UTC", size=12, anchor="middle")
    text(80, 413, "Adaptive recovered-label audit; different population and training protocol from Probe CV.", size=13)
    parts.append('</svg>')
    return "\n".join(parts)


def main():
    items = []
    for ident, score, time, strategy, metric, credit, status, title, note, source, basis in MILESTONES:
        if not (ROOT / source).exists():
            raise FileNotFoundError(source)
        items.append(dict(id=ident, score=float(score), time=time, strategy=strategy,
                          metric=metric, credit=credit, creditCommit=RECORD_COMMITS.get(ident), status=status, title=title,
                          note=note, source=source, basis=basis))
    frontier = [item["score"] for item in items if item["metric"] == "cv" and item["status"] == "promoted"]
    if any(before <= after for before, after in zip(frontier, frontier[1:])):
        raise ValueError("Main journey must descend; document screens separately")
    if any(before["score"] <= after["score"] for before, after in zip(COMPETITION, COMPETITION[1:])):
        raise ValueError("Selected competition leaderboard milestones must descend")
    payload = dict(milestones=items, runs=[run_curve(*run) for run in RUNS], competition=COMPETITION)
    template = (ROOT / "research_journey.template.html").read_text()
    marker = "__JOURNEY_DATA__"
    if template.count(marker) != 1:
        raise ValueError("Expected one data insertion marker")
    output = template.replace(marker, json.dumps(payload, separators=(",", ":")))
    output = output.replace("__QUALIFYING_GRAPH__", qualifying_svg())
    (ROOT / "assets/qualifying_progress.svg").write_text(qualifying_svg() + "\n")
    for name, filename, color in (("claude", "claude-code.svg", "#D97757"), ("codex", "codex.svg", "#b6a5ff")):
        icon = (ROOT / "assets" / filename).read_text().replace('fill="currentColor"', f'fill="{color}"')
        data_url = "data:image/svg+xml;base64," + base64.b64encode(icon.encode()).decode()
        output = output.replace(f"__{name.upper()}_LOGO__", data_url)
    (ROOT / "research_journey.html").write_text(output)
    print(f"research_journey.html: {len(items)} milestones, {len(payload['runs'])} runtime curves")


if __name__ == "__main__":
    main()
