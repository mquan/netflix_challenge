#!/usr/bin/env python3
"""Admit the complete own CF96 trajectory without fitting or outcome selection."""
import hashlib
import re
from pathlib import Path
import numpy as np
import upstream_cfnade_parallel as native
import upstream_remote_cf96 as checkpoint

u = native.u
REPORT = u.ROOT / 'experiments/upstream-cf96-complete-bank'
DECL = u.ROOT / 'CF96_COMPLETE_BANK_PLAN.md'
NAMES = [f'cfnade-96_ep{epoch:02}' for epoch in range(1, 96)] + ['cfnade-96', 'cfnade-96r', 'cfnade-96r_ep95']
TRACES = {
    'probex': ['upstream-cfnade-parallel/x-attempt-1795029414544754000.log',
               'upstream-cf96-disk-resume/before-resume-probex.out'],
    'qual': ['upstream-cfnade-parallel/qual-attempt-1795029414546664000.log',
             'upstream-cf96-disk-resume/before-resume-qual.out',
             'upstream-cf96-disk-resume/qual-attempt-1795152685218313000.log'],
}


def clipped_sha(values):
    digest = hashlib.sha256()
    for lo in range(0, len(values), 65536):
        digest.update(np.clip(values[lo:lo+65536], 0, 6).astype('<f4').tobytes())
    return digest.hexdigest()


def lineage():
    native.common.verify_candidate()
    plan_path = native.REPORT / 'plan.json'
    receipt_path = native.REPORT / 'native-receipt.json'
    completion_path = u.ROOT / 'experiments/upstream-cf96-disk-resume/recovery-completion.json'
    plan, receipt, completion = map(u.read, [plan_path, receipt_path, completion_path])
    if (completion['state'] != 'complete_exact95_95_and_original_finalizer' or
        completion['original_receipt_sha256'] != u.sha(receipt_path) or
        receipt['source_commit'] != u.PIN or receipt['job'] != 'cfnade-96' or
        plan['source_pin'] != u.PIN or
        plan['source_id'] != u.sha(native.candidate.REPORT/'preparation.json') or
        plan['proof_sha256'] != u.sha(native.candidate.REPORT/'verification.json') or
        receipt['plan_sha256'] != u.sha(plan_path) or
        plan['epochs_per_phase'] != 95 or plan['hidden'] != 320 or plan['rank'] != 96):
        raise ValueError('Completed original CF96 lineage required')
    bindings = dict(plan['input_sha256'])
    import json
    if hashlib.sha256(json.dumps(bindings, sort_keys=True).encode()).hexdigest() != plan['data_id']:
        raise ValueError('Original dataset identity differs')
    for name, digest in plan['binaries'].items():
        bindings[str((native.BASE/'target/release'/name).relative_to(u.ROOT))] = digest
    paths = [plan_path, receipt_path, completion_path, DECL, Path(__file__),
             u.ROOT/'scripts/verify_upstream_cf96_complete_bank.py',
             native.candidate.REPORT/'preparation.json', native.candidate.REPORT/'verification.json']
    for relative in u.read(native.candidate.REPORT/'preparation.json')['files']:
        paths.append(native.TREE/relative)
    phases = {}
    for train, (query, rows) in checkpoint.PHASES.items():
        base = native.BASE/'parallel-checkpoints'/train
        pointer, state, directory = checkpoint.phase_checkpoint(base, plan['source_id'], plan['data_id'], query, rows)
        if pointer['epoch'] != 95 or state['identity']['train'] != train:
            raise ValueError('Both full95 original checkpoints required')
        paths += [base/'owner.json', base/'latest.json', directory/'state.json']
        paths += [directory/name for name in state['files']]
        traces = [u.ROOT/'experiments'/name for name in TRACES[query]]
        traces.append(native.TREE/'preds_new'/f'cfnade-96.phase-{query}.out')
        epochs = set()
        for trace in traces:
            segment = [int(i) for i in re.findall(r'^Epoch (\d+) \|', trace.read_text(), re.M)]
            if not segment or segment != list(range(segment[0], segment[-1]+1)):
                raise ValueError('Original resume trace must contain contiguous epochs')
            epochs.update(segment)
        if epochs != set(range(1, 96)):
            raise ValueError('Complete original95 epoch trace required')
        paths.extend(traces)
        phases[train] = dict(epoch=95, tensors=84, state_sha256=u.sha(directory/'state.json'))
    for path in paths:
        bindings[str(path.relative_to(u.ROOT))] = u.sha(path)
    for relative, digest in bindings.items():
        if u.sha(u.ROOT/relative) != digest:
            raise ValueError('Original source/data/binary/checkpoint changed: '+relative)
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy', mmap_mode='r')):
        raise ValueError('Qualifying training targets must be zero')
    return receipt, bindings, phases


def main():
    if REPORT.exists():
        raise ValueError('Existing admission evidence preserved')
    receipt, bindings, phases = lineage()
    REPORT.mkdir()
    u.write(REPORT/'plan.json', dict(names=NAMES, source_sha256=bindings,
        phases=phases, qualifying_labels_read=False, models_fitted=0))
    rows, aliases, seen, all_rows = [], [], {}, []
    for name in NAMES:
        pair, digests = {}, []
        for query, count in [('probex',1128164), ('qual',2817131)]:
            relative = 'preds_new/'+name+'.'+query+'.npy'
            path = native.TREE/relative
            digest = receipt['outputs'][relative]
            if u.sha(path) != digest:
                raise ValueError('Original query receipt mismatch: '+relative)
            values = np.load(path, mmap_mode='r', allow_pickle=False)
            if values.shape != (count,) or values.dtype != np.dtype('<f4') or not np.isfinite(values).all():
                raise ValueError('Full finite native float32 query required')
            pair[query] = str(path.relative_to(u.ROOT))
            bindings[pair[query]] = digest
            digests.append(clipped_sha(values))
        row = dict(name=name, paired_files=pair, clipped_pair_sha256=digests)
        all_rows.append(row)
        key = tuple(digests)
        if key in seen:
            aliases.append(dict(name=name, alias_of=seen[key], clipped_pair_sha256=digests))
        else:
            seen[key] = name
            rows.append(row)
    for relative, digest in bindings.items():
        if u.sha(u.ROOT/relative) != digest:
            raise ValueError('Input changed during admission')
    u.write(REPORT/'manifest.json', dict(state='complete_all98_cf96_pairs_pending_independent_replay',
        rows=rows, all_rows=all_rows, aliases=aliases, source_sha256=bindings,
        plan_sha256=u.sha(REPORT/'plan.json'), qualifying_labels_read=False, models_fitted=0,
        prediction_inference_replayed=False, public_prediction_values_read=False))
    print(f'Admitted {len(all_rows)} pairs, {len(rows)} distinct; independent replay remains', flush=True)


if __name__ == '__main__':
    main()
