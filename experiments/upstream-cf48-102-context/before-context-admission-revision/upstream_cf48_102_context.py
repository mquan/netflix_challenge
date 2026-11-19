#!/usr/bin/env python3
"""Fixed rank16 context NN on own102 with the published CF48 epoch25 snapshot."""
import argparse
import math
from pathlib import Path
import subprocess
import numpy as np
import upstream_extension_context as base
import upstream_cf48_102 as data

u,n=base.u,base.n
BASE_FILE=Path(base.__file__)
PREVIOUS_REPORT,PREVIOUS_RUN=base.REPORT,base.RUN
ORIGINAL_HASHES=base.code_hashes
ORIGINAL_PREDECESSORS=base.predecessors
RUN=u.ROOT/'artifacts/upstream-cf48-102-context'
REPORT=u.ROOT/'experiments/upstream-cf48-102-context'
CONFIG=dict(base.CONFIG,architecture='context-weighted-residual-cf48-102-v1')
SIGNALS=list(range(63))+list(range(68,91))+list(range(221,239))
CONTEXT=list(base.CONTEXT)


def model_factory(width,cfg):
    import torch
    if width!=239 or cfg!=CONFIG:raise ValueError('Frozen own102 context recipe required')
    class ContextResidual(torch.nn.Module):
        def __init__(self):
            super().__init__();h,r,d=cfg['nn_hidden'],cfg['context_rank'],cfg['nn_dropout']
            self.register_buffer('signals',torch.tensor(SIGNALS));self.register_buffer('context',torch.tensor(CONTEXT))
            self.base=torch.nn.Sequential(torch.nn.Linear(width,h),torch.nn.GELU(),torch.nn.Dropout(d),
                torch.nn.Linear(h,h),torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,1))
            self.project=torch.nn.Linear(len(SIGNALS),r,bias=False)
            self.gate=torch.nn.Sequential(torch.nn.Linear(len(CONTEXT),h),torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,r))
            for layer in [self.base[-1],self.gate[-1]]:
                torch.nn.init.zeros_(layer.weight);torch.nn.init.zeros_(layer.bias)
        def forward(self,values):
            factors=self.project(values.index_select(1,self.signals));weights=self.gate(values.index_select(1,self.context))
            return self.base(values)+(factors*weights).sum(1,keepdim=True)/math.sqrt(cfg['context_rank'])
    return ContextResidual()


def code_hashes():
    return {**ORIGINAL_HASHES(),**data.code_hashes(),str(BASE_FILE.relative_to(u.ROOT)):u.sha(BASE_FILE),
        str(Path(__file__).relative_to(u.ROOT)):u.sha(Path(__file__)),
        'scripts/verify_upstream_cf48_prefix_screen.py':u.sha(u.ROOT/'scripts/verify_upstream_cf48_prefix_screen.py')}


def predecessors():
    found=ORIGINAL_PREDECESSORS()
    for row in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        pid,command=row.strip().split(None,1)
        if any(Path(arg).name==BASE_FILE.name for arg in command.split()[1:]):found[pid]=command
    return found


def configure():
    # Rebind only this interpreter's controller. Existing source and experiments
    # remain unchanged; all reused implementation files are hash-bound.
    base.RUN,base.REPORT,base.CONFIG=RUN,REPORT,CONFIG
    base.SIGNALS,base.CONTEXT=SIGNALS,CONTEXT
    base.model_factory,base.code_hashes,base.prepare,base.predecessors=model_factory,code_hashes,prepare,predecessors
    base.__file__=str(Path(__file__).resolve())
    base.configure()


def prepare():
    configure()
    if (REPORT/'manifest.json').exists():return n.verify_prepared()
    if RUN.exists():raise ValueError('Partial own102 context cache preserved')
    dm=u.read(data.REPORT/'manifest.json')
    n.RUN,n.REPORT,n.CONFIG,n.code_hashes=data.RUN,data.REPORT,dm['config'],data.code_hashes
    n.verify_prepared()
    ridge_proof=data.REPORT/'cache-column-verification.json';checked=u.read(ridge_proof)
    independent_ridge=data.REPORT/'ridge-verification.json'
    if (checked['state']!='complete_cache_columns' or not checked['outer_users_disjoint'] or
            checked['manifest_sha256']!=u.sha(data.REPORT/'manifest.json') or
            checked['screen_sha256']!=u.sha(data.REPORT/'ridge-screen.json') or
            checked['source_ridge_verification_sha256']!=u.sha(independent_ridge) or
            dm['new_native_predictions']!=data.NEW or dm['prediction_columns']!=102):
        raise ValueError('Verified fixed102 inputs required')
    source=base.source;sm=u.read(source.REPORT/'manifest.json');evaluation=source.REPORT/'evaluation.json'
    proof=source.REPORT/'independent-verification.json';replay=source.REPORT/'tree-replay.json'
    if (u.read(proof)['evaluation_sha256']!=u.sha(evaluation) or
            not u.read(proof)['fixed_average_bitwise_equal'] or
            [r['fold'] for r in u.read(replay)['records']]!=list(range(5)) or
            not all(r['bitwise_equal'] for r in u.read(replay)['records'])):
        raise ValueError('Complete reused-tree verification required')
    sources={str(p.relative_to(u.ROOT)):u.sha(p) for p in
        [data.REPORT/'manifest.json',ridge_proof,independent_ridge,data.REPORT/'ridge-screen.json',source.REPORT/'manifest.json',evaluation,proof,replay]}
    for key in ['labels.npy','users.npy','groups.npy']:
        if dm['prepared_sha256'][key]!=sm['prepared_sha256'][key]:raise ValueError('Own102/reused-tree identities differ')
    for fold in range(5):
        path=source.REPORT/'tree'/f'{fold}.json';receipt=u.read(path)
        if receipt['manifest_sha256']!=u.sha(source.REPORT/'manifest.json'):raise ValueError('Reused tree receipt differs')
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
        for name,digest in receipt['outputs'].items():
            artifact=source.RUN/'tree'/str(fold)/name
            if u.sha(artifact)!=digest:raise ValueError('Actual reused tree model changed')
            sources[str(artifact.relative_to(u.ROOT))]=digest
    tree=next(r for r in u.read(evaluation)['records'] if r['arm']=='tree')
    if u.sha(source.RUN/'tree.f32')!=tree['prediction_sha256']:raise ValueError('Reused tree vector differs')
    combined_path=u.ROOT/'experiments/upstream-tree91-context86/verification.json';combined=u.read(combined_path)
    if combined['qualifying_labels_read'] is not False or not combined['fixed_average_bitwise_equal']:
        raise ValueError('Known-label global reference required')
    for relative,digest in combined['source_sha256'].items():
        if u.sha(u.ROOT/relative)!=digest:raise ValueError('Global reference source changed')
        sources[relative]=digest
    sources[str(combined_path.relative_to(u.ROOT))]=u.sha(combined_path);reference=combined['rmse']
    previous_context=None;previous_proof=PREVIOUS_REPORT/'independent-verification.json'
    if previous_proof.exists():
        previous_evaluation=PREVIOUS_REPORT/'evaluation.json';old=u.read(previous_proof)
        if old['state']!='complete' or old['evaluation_sha256']!=u.sha(previous_evaluation):
            raise ValueError('Previous context result binding differs')
        previous_context=next(r for r in old['records'] if r['arm']=='fixed-half-tree-nn')
        vector=PREVIOUS_RUN/'fixed-half-tree-nn.f32'
        if u.sha(vector)!=previous_context['prediction_sha256']:raise ValueError('Previous context vector differs')
        for path in [previous_proof,previous_evaluation,vector]:sources[str(path.relative_to(u.ROOT))]=u.sha(path)
        reference=min(reference,previous_context['independent_rmse'])
    configure();RUN.mkdir();REPORT.mkdir(parents=True,exist_ok=True)
    for name,digest in dm['prepared_sha256'].items():
        u.link_or_copy(data.RUN/name,RUN/name)
        if u.sha(RUN/name)!=digest:raise ValueError('Copied own102 input differs')
    u.link_or_copy(source.RUN/'tree.f32',RUN/'tree.f32')
    manifest=dict(config=CONFIG,versions=n.versions(),code_sha256=code_hashes(),source_commit=u.PIN,
        source_manifests_sha256=sources,prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},
        rows=dm['rows'],baseline_rmse=dm['baseline_rmse'],prediction_columns=102,design_columns=178,nonlinear_features=238,
        signal_indices=SIGNALS,context_indices=CONTEXT,previous_best_rmse=reference,
        previous_context91=previous_context,reused_tree_prediction_sha256=tree['prediction_sha256'],
        full95_model_complete=False,cf48_feature_epoch=25,cf48_original_configured_epochs=95,
        arms=['control','tree','nn','fixed-half-tree-nn'],qualifying_labels_read=False,isolation=dm['isolation'],
        scope='Adaptive known-ProbeX only. Fixed original rank16 context architecture plus ready-eight and own CF48 epoch25 additions, same five nested user folds/seeds/optimizer/stopping. Reuses actual verified tree91, without synthetic fitting receipts. No production or qualifying audit launched.')
    u.write(REPORT/'manifest.json',manifest);return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');p.add_argument('--replay',action='store_true');a=p.parse_args();configure()
    if a.worker:n.worker('nn')
    elif a.replay:base.verifier.replay('nn')
    else:
        try:base.run()
        except BaseException as error:
            if not isinstance(error,BlockingIOError):u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)))
            raise
