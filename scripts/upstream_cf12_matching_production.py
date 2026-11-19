#!/usr/bin/env python3
"""Production for the verified86-tree/86-context-NN development average."""
import argparse
import datetime as dt
from pathlib import Path
import threading
from types import SimpleNamespace
import numpy as np
import upstream_cf12_production as previous
import upstream_cf12_tree as development

u,n,core,bank=previous.u,previous.n,previous.core,previous.bank
CORE_FILE=previous.CORE_FILE
ORIGINAL_WORKER=previous.ORIGINAL_WORKER
RUN=u.ROOT/'artifacts/upstream-cf12-matching-production'
REPORT=u.ROOT/'experiments/upstream-cf12-matching-production'


def code_hashes():
    paths=[Path(__file__),Path(previous.__file__),CORE_FILE,
        u.ROOT/'scripts/verify_upstream_cf12_tree_comparison.py',
        Path(previous.extension.__file__),Path(previous.previous_queue.__file__)]
    return {**development.code_hashes(),**bank.code_hashes(),
        **{str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}}


def development_gate():
    development.configure();n.verify_prepared()
    report=development.REPORT;old=development.parent.REPORT
    evaluation=u.read(report/'evaluation.json');proof=u.read(report/'independent-verification.json')
    comparison=u.read(report/'comparison-previous.json')
    if (proof['state']!='complete' or proof['evaluation_sha256']!=u.sha(report/'evaluation.json') or
            not proof['fixed_average_bitwise_equal'] or comparison['state']!='verified_complete_comparison' or
            not comparison['neural_vector_unchanged'] or not comparison['all_tree_and_neural_artifacts_rehashed'] or
            comparison['qualifying_labels_read'] is not False):
        raise ValueError('Complete independently verified matching development required')
    for relative,digest in comparison['source_sha256'].items():
        if u.sha(u.ROOT/relative)!=digest:raise ValueError('Matching comparison source changed')
    if comparison['verifier_sha256']!=u.sha(u.ROOT/'scripts/verify_upstream_cf12_tree_comparison.py'):
        raise ValueError('Matching comparison verifier changed')
    for path in [report/'tree-replay.json',old/'nn-replay.json']:
        replay=u.read(path)
        if [r['fold'] for r in replay['records']]!=list(range(5)) or not all(r['bitwise_equal'] for r in replay['records']):
            raise ValueError('Complete tree and neural replay required')
    row=next(r for r in evaluation['records'] if r['arm']=='fixed-half-tree-nn')
    score=comparison['independent_rmse']['matching']
    if (abs(score-row['rmse'])>1e-12 or not score<comparison['independent_rmse']['previous'] or
            u.sha(development.RUN/'fixed-half-tree-nn.f32')!=row['prediction_sha256']):
        raise ValueError('Matching candidate must improve the complete preceding score')
    return score


def budget_sources(origins=None):
    origins=origins or [('tree',development.REPORT,development.RUN),
                       ('nn',development.parent.REPORT,development.parent.RUN)]
    choices={};sources={}
    for family,report,cache in origins:
        manifest=report/'manifest.json'
        if u.read(manifest)['config']!=n.CONFIG:raise ValueError('Budget origin recipe differs')
        sources[str(manifest.relative_to(u.ROOT))]=u.sha(manifest);records=[]
        for fold in range(5):
            path=report/family/(str(fold)+'.json');receipt=u.read(path)
            if (receipt['manifest_sha256']!=u.sha(manifest) or receipt['family']!=family or receipt['fold']!=fold):
                raise ValueError('Actual originating family/fold receipt required')
            for name,digest in receipt['outputs'].items():
                model=cache/family/str(fold)/name
                if u.sha(model)!=digest:raise ValueError('Budget-source model artifact changed')
                sources[str(model.relative_to(u.ROOT))]=digest
            sources[str(path.relative_to(u.ROOT))]=u.sha(path);records.append(receipt['chosen_iterations'])
        choices[family]=core.budgets(records,n.CONFIG['tree_rounds' if family=='tree' else 'nn_epochs'])
    if set(choices)!={'tree','nn'}:raise ValueError('Both actual budget origins required')
    return choices,sources


def prepare():
    if (REPORT/'plan.json').exists():return core.verify()
    if RUN.exists():raise ValueError('Partial matching production preparation preserved')
    candidate=development_gate();data=bank.build();development.configure()
    if (data['prediction_columns'],data['design_columns'],data['nonlinear_features'])!=(86,162,222):
        raise ValueError('Both matching families require the86-predictor bank')
    choices,sources=budget_sources();sources.update(data['source_sha256'])
    for report,names in [(development.REPORT,['manifest.json','evaluation.json','independent-verification.json','tree-replay.json','comparison-previous.json']),
                         (development.parent.REPORT,['manifest.json','evaluation.json','independent-verification.json','nn-replay.json']),
                         (bank.REPORT,['manifest.json'])]:
        for name in names:
            p=report/name;sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in data['files'].items():sources[str((bank.RUN/name).relative_to(u.ROOT))]=digest
    RUN.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
    z=np.load(bank.RUN/'probex-design.npy',mmap_mode='r');y=np.load(bank.RUN/'labels.npy').astype(np.float64)
    if z.shape!=(data['rows'],162) or y.shape!=(data['rows'],):raise ValueError('Matching known-row layout differs')
    gram=np.zeros((162,162));rhs=np.zeros(162)
    for lo in range(0,len(y),8192):
        hi=min(lo+8192,len(y));a=z[lo:hi];gram+=a.T@a;rhs+=a.T@y[lo:hi]
    weights=n.ridge_weights(gram[None],rhs[None],np.array([True]),n.CONFIG['ridge_lambda'])
    np.save(RUN/'ridge.npy',weights)
    plan=dict(state='frozen_before_final_fitting',created=dt.datetime.now(dt.timezone.utc).isoformat(),
        config=n.CONFIG,versions=n.versions(),code_sha256=code_hashes(),source_sha256=sources,
        iterations=choices,seeds=n.CONFIG['seeds'],qualifying_rows=data['qualifying_rows'],qualifying_labels_read=False,
        ridge_sha256=u.sha(RUN/'ridge.npy'),prediction_chunk_rows=core.CHUNK,development_rmse=candidate,
        iteration_rule='Per-seed median of five actual matching-tree receipts and five actual preceding context-NN receipts; no synthetic neural fitting receipts.',
        bank_scope_note='Read-only reuse of the paired86 bank originally prepared for the earlier mixed85-tree/86-NN consumer. Both families in this new run consume its162-column ridge and222 features plus ridge input; no85-tree output is reused.',
        final_rule='Fresh full-known-ProbeX refits of both86-predictor families. Original context neural architecture and tree recipe; three seed means, family clipping and fixed equal float32-family average. Saved-model replay, freeze, then independent recovered-group audit.')
    u.write(REPORT/'plan.json',plan);return plan


def configure():
    development.configure()
    core.RUN,core.REPORT,core.bank,core.voting=RUN,REPORT,bank,development
    core.code_hashes,core.prepare,core.worker=code_hashes,prepare,ORIGINAL_WORKER
    core.__file__=str(Path(__file__).resolve())


def relay(stopping):
    while not stopping.wait(5):
        path=REPORT/'status.json'
        if path.exists():
            state=u.read(path);state['scheduler']='cf12-matching-production';u.write(previous.OPERATIONAL,state)


def run_queued():
    # Reuse the unchanged, tested first-audit gate, shared fitting lock, memory
    # admission and stronger91-result check only in this interpreter.
    previous.RUN,previous.REPORT=RUN,REPORT
    previous.development_gate,previous.relay=development_gate,relay
    previous.development=SimpleNamespace(resources=development.parent.resources)
    previous.run_queued()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--family',choices=['tree','nn']);parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();configure()
    if args.family:ORIGINAL_WORKER(args.family,args.replay)
    else:run_queued()
