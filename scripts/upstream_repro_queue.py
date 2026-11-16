#!/usr/bin/env python3
"""Run the eight first fresh upstream jobs sequentially, comparing and committing after each.

Each job goes through the existing wrapper (scripts/upstream_repro.py --run-job),
so receipts, hash checks and refusal of unreceipted outputs are unchanged. After
every completed job this script:
  1. runs --compare-model for it against the released ProbeX prediction,
  2. copies all execution receipts, log tails and comparisons into the tracked
     experiments/upstream-repro/ directory,
  3. regenerates UPSTREAM_PROGRESS.md (per-job scores and residual-chain gains),
  4. commits and pushes to research/upstream-reproduction.
Already-receipted jobs are skipped by the wrapper, so rerunning resumes.
"""
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/upstream-repro'
TRACKED = ROOT/'experiments/upstream-repro'
BRANCH = 'research/upstream-reproduction'
PY = sys.executable
JOBS = ['tsvdx4-60', 'tsvdx4-60__knns',
        'rbmx2-40', 'rbmx2-40__knn3',
        'dnn-24', 'dnn-24__dnn-16p50', 'dnn-24__dnn-16p50__knn3',
        'dnn-24__knn3']
CHAINS = [('tsvdx4-60', 'tsvdx4-60__knns'), ('rbmx2-40', 'rbmx2-40__knn3'),
          ('dnn-24', 'dnn-24__knn3'), ('dnn-24', 'dnn-24__dnn-16p50'),
          ('dnn-24__dnn-16p50', 'dnn-24__dnn-16p50__knn3')]
STATE = BASE/'queue-state.json'


def log(msg):
    print(f'{dt.datetime.now().strftime("%H:%M:%S")} {msg}', flush=True)


def read(path):
    return json.loads(Path(path).read_text())


def sync_tracked():
    receipts, comparisons = TRACKED/'fresh-receipts', TRACKED/'comparisons'
    receipts.mkdir(parents=True, exist_ok=True)
    comparisons.mkdir(parents=True, exist_ok=True)
    for p in sorted((BASE/'fresh-receipts').glob('*.json')):
        shutil.copyfile(p, receipts/p.name)
        lg = p.with_suffix('.log')
        if lg.exists():
            lines = lg.read_text(errors='replace').splitlines()
            (receipts/(p.stem+'.log.tail')).write_text('\n'.join(lines[-60:])+'\n')
    for p in sorted((BASE/'comparisons').glob('*.json')) if (BASE/'comparisons').exists() else []:
        shutil.copyfile(p, comparisons/p.name)
    for name in ['fresh-data-receipt.json', 'build-receipt.json', 'fresh-training-plan.json']:
        if (BASE/name).exists():
            shutil.copyfile(BASE/name, TRACKED/name)


def hours(receipt):
    r = read(receipt)
    t = dt.datetime.fromisoformat(r['finished'])-dt.datetime.fromisoformat(r['started'])
    return t.total_seconds()/3600


def render(state):
    now = dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')
    out = ['# Upstream fresh-training progress (M3 Max, 96 GB)', '',
           f'Auto-updated by `scripts/upstream_repro_queue.py`. Last update: {now}.', '',
           f'- Queue state: **{state["state"]}**' + (f' — running `{state["current"]}`' if state.get('current') else ''),
           f'- Threads: RAYON {os.environ.get("UPSTREAM_REPRO_THREADS", "2")}, BLAS {os.environ.get("UPSTREAM_REPRO_BLAS_THREADS", "1")}']
    if state.get('error'):
        out.append(f'- Error: `{state["error"]}`')
    out += ['', 'ProbeX development diagnostics only: fresh training, released vectors imported for comparison.', '',
            '| Job | State | Train h | Fresh RMSE | Released RMSE | Fresh − released | Pred RMS diff | Max abs diff |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    cmp = {}
    for j in JOBS:
        r = BASE/'fresh-receipts'/(j+'.json')
        c = BASE/'comparisons'/(j+'.json')
        st = 'done' if r.exists() else 'running' if state.get('current') == j else 'queued'
        row = [j, st, f'{hours(r):.2f}' if r.exists() else '']
        if c.exists():
            d = cmp[j] = read(c)
            row += [f'{d["fresh_rmse"]:.6f}', f'{d["released_rmse"]:.6f}',
                    f'{d["fresh_rmse"]-d["released_rmse"]:+.6f}',
                    f'{d["prediction_rmse_difference"]:.6f}', f'{d["max_absolute_difference"]:.4f}']
        else:
            row += ['']*5
        out.append('| '+' | '.join(row)+' |')
    out += ['', '## Residual-chain gains (RMSE reduction from parent to child)', '',
            '| Parent → child | Fresh gain | Released gain |', '|---|---:|---:|']
    for a, b in CHAINS:
        if a in cmp and b in cmp:
            out.append(f'| {a} → {b} | {cmp[a]["fresh_rmse"]-cmp[b]["fresh_rmse"]:.6f} | '
                       f'{cmp[a]["released_rmse"]-cmp[b]["released_rmse"]:.6f} |')
        else:
            out.append(f'| {a} → {b} | pending | pending |')
    extra = sorted(p.stem for p in (BASE/'fresh-receipts').glob('*.json') if p.stem not in JOBS) \
        if (BASE/'fresh-receipts').exists() else []
    if extra:
        out += ['', 'Dependency jobs also executed (receipts committed): ' +
                ', '.join(f'`{e}` ({hours(BASE/"fresh-receipts"/(e+".json")):.2f} h)' for e in extra)]
    (ROOT/'UPSTREAM_PROGRESS.md').write_text('\n'.join(out)+'\n')


def commit(message, state):
    sync_tracked()
    render(state)
    STATE.write_text(json.dumps(state, indent=2))
    subprocess.run(['git', 'add', 'UPSTREAM_PROGRESS.md', 'experiments/upstream-repro'], cwd=ROOT, check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode == 0:
        return
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    subprocess.run(['git', '-c', 'user.email=mquannie@gmail.com', 'commit', '-q', '-m',
                    message+'\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'], cwd=ROOT, check=True)
    ok = subprocess.run(['git', 'pull', '-q', '--rebase', 'origin', BRANCH], cwd=ROOT, env=env).returncode == 0 and \
        subprocess.run(['git', 'push', '-q', 'origin', 'HEAD:'+BRANCH], cwd=ROOT, env=env).returncode == 0
    log(('pushed: ' if ok else 'PUSH FAILED (will retry next commit): ')+message)


def main():
    state = dict(state='running', current=None, completed=[])
    for job in JOBS:
        if (BASE/'comparisons'/(job+'.json')).exists():
            state['completed'].append(job)
            continue
        state['current'] = job
        commit(f'upstream-repro progress: start {job}', state)
        log(f'start {job}')
        run = subprocess.run([PY, 'scripts/upstream_repro.py', '--run-job', job], cwd=ROOT)
        if run.returncode:
            state.update(state='failed', error=f'{job} --run-job exited {run.returncode}; see artifacts/upstream-repro/fresh-receipts/*.log')
            commit(f'upstream-repro: {job} FAILED', state)
            sys.exit(run.returncode)
        cmp = subprocess.run([PY, 'scripts/upstream_repro.py', '--compare-model', job], cwd=ROOT)
        if cmp.returncode:
            state.update(state='failed', error=f'{job} --compare-model exited {cmp.returncode}')
            commit(f'upstream-repro: {job} comparison FAILED', state)
            sys.exit(cmp.returncode)
        d = read(BASE/'comparisons'/(job+'.json'))
        state['completed'].append(job)
        state['current'] = None
        commit(f'upstream-repro: {job} done, fresh {d["fresh_rmse"]:.6f} vs released {d["released_rmse"]:.6f}', state)
        log(f'done {job}')
    state.update(state='complete', current=None)
    commit('upstream-repro: all eight fresh jobs complete', state)


if __name__ == '__main__':
    main()
